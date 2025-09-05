import os
import multiprocessing
from collections import OrderedDict
from time import time

import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *  # noqa: F401,F403
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as sql
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer, VectorSlicer
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier


COL_NAMES = np.array([
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"
])

NOMINAL_INX = [1, 2, 3]
BINARY_INX = [6, 11, 13, 14, 20, 21]
NUMERIC_INX = list(set(range(41)).difference(NOMINAL_INX).difference(BINARY_INX))

NOMINAL_COLS = COL_NAMES[NOMINAL_INX].tolist()
BINARY_COLS = COL_NAMES[BINARY_INX].tolist()
NUMERIC_COLS = COL_NAMES[NUMERIC_INX].tolist()

LABELS2 = ['normal', 'attack']

SEED = 4667979835606274383
KMEANS_K = 8
KMEANS_MAX_ITER = 100
KMEANS_INIT_STEPS = 25
AR_THRESHOLD_FOR_ASSEMBLER = 0.01
AR_THRESHOLD_FOR_KMEANS_NUMERIC = 0.1
AR_THRESHOLD_FOR_RF = 0.05
THRESHOLD_TEST = 0.01


attack_dict = {
    'normal': 'normal',
    'back': 'DoS','land': 'DoS','neptune': 'DoS','pod': 'DoS','smurf': 'DoS','teardrop': 'DoS','mailbomb': 'DoS','apache2': 'DoS','processtable': 'DoS','udpstorm': 'DoS',
    'ipsweep': 'Probe','nmap': 'Probe','portsweep': 'Probe','satan': 'Probe','mscan': 'Probe','saint': 'Probe',
    'ftp_write': 'R2L','guess_passwd': 'R2L','imap': 'R2L','multihop': 'R2L','phf': 'R2L','spy': 'R2L','warezclient': 'R2L','warezmaster': 'R2L','sendmail': 'R2L','named': 'R2L','snmpgetattack': 'R2L','snmpguess': 'R2L','xlock': 'R2L','xsnoop': 'R2L','worm': 'R2L',
    'buffer_overflow': 'U2R','loadmodule': 'U2R','perl': 'U2R','rootkit': 'U2R','httptunnel': 'U2R','ps': 'U2R','sqlattack': 'U2R','xterm': 'U2R'
}
attack_mapping_udf = udf(lambda v: attack_dict.get(v, 'attack'))


class Labels2Converter(Transformer):
    def __init__(self):
        super(Labels2Converter, self).__init__()

    def _transform(self, dataset):
        return dataset.withColumn('labels2', sql.regexp_replace(col('labels'), '^(?!normal).*$', 'attack'))


class Labels5Converter(Transformer):
    def __init__(self):
        super(Labels5Converter, self).__init__()

    def _transform(self, dataset):
        return dataset.withColumn('labels5', attack_mapping_udf(col('labels')))


def build_spark():
    conf = (
        SparkConf()
        .setMaster(f"local[{multiprocessing.cpu_count()}]")
        .setAppName("PySpark NSL-KDD - Web")
        .setAll([
            ("spark.driver.memory", "6g"),
            ("spark.default.parallelism", f"{multiprocessing.cpu_count()}"),
        ])
    )
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("WARN")
    sql_ctx = SQLContext(sc)
    return sc, sql_ctx


def load_dataset(sc, path):
    rdd = sc.textFile(path, 8).map(lambda line: line.split(','))
    df = (
        rdd.toDF(COL_NAMES.tolist()).select(
            col('duration').cast(DoubleType()),
            col('protocol_type').cast(StringType()),
            col('service').cast(StringType()),
            col('flag').cast(StringType()),
            col('src_bytes').cast(DoubleType()),
            col('dst_bytes').cast(DoubleType()),
            col('land').cast(DoubleType()),
            col('wrong_fragment').cast(DoubleType()),
            col('urgent').cast(DoubleType()),
            col('hot').cast(DoubleType()),
            col('num_failed_logins').cast(DoubleType()),
            col('logged_in').cast(DoubleType()),
            col('num_compromised').cast(DoubleType()),
            col('root_shell').cast(DoubleType()),
            col('su_attempted').cast(DoubleType()),
            col('num_root').cast(DoubleType()),
            col('num_file_creations').cast(DoubleType()),
            col('num_shells').cast(DoubleType()),
            col('num_access_files').cast(DoubleType()),
            col('num_outbound_cmds').cast(DoubleType()),
            col('is_host_login').cast(DoubleType()),
            col('is_guest_login').cast(DoubleType()),
            col('count').cast(DoubleType()),
            col('srv_count').cast(DoubleType()),
            col('serror_rate').cast(DoubleType()),
            col('srv_serror_rate').cast(DoubleType()),
            col('rerror_rate').cast(DoubleType()),
            col('srv_rerror_rate').cast(DoubleType()),
            col('same_srv_rate').cast(DoubleType()),
            col('diff_srv_rate').cast(DoubleType()),
            col('srv_diff_host_rate').cast(DoubleType()),
            col('dst_host_count').cast(DoubleType()),
            col('dst_host_srv_count').cast(DoubleType()),
            col('dst_host_same_srv_rate').cast(DoubleType()),
            col('dst_host_diff_srv_rate').cast(DoubleType()),
            col('dst_host_same_src_port_rate').cast(DoubleType()),
            col('dst_host_srv_diff_host_rate').cast(DoubleType()),
            col('dst_host_serror_rate').cast(DoubleType()),
            col('dst_host_srv_serror_rate').cast(DoubleType()),
            col('dst_host_rerror_rate').cast(DoubleType()),
            col('dst_host_srv_rerror_rate').cast(DoubleType()),
            col('labels').cast(StringType()),
        )
    )
    return df


def ohe_vec(cat_dict, value):
    vec = np.zeros(len(cat_dict))
    vec[cat_dict[value]] = float(1.0)
    return vec.tolist()


def ohe(df, nominal_col):
    categories = (
        df.select(nominal_col).distinct().rdd.map(lambda row: row[0]).collect()
    )
    cat_dict = dict(zip(categories, range(len(categories))))
    schema = StructType([StructField(cat, DoubleType(), False) for cat in categories])
    udf_ohe_vec = udf(lambda v: ohe_vec(cat_dict, v), schema)
    df = df.withColumn(nominal_col + '_ohe', udf_ohe_vec(col(nominal_col))).cache()
    nested_cols = [nominal_col + '_ohe.' + cat for cat in categories]
    ohe_cols = [nominal_col + '_' + cat for cat in categories]
    for new, old in zip(ohe_cols, nested_cols):
        df = df.withColumn(new, col(old))
    df = df.drop(nominal_col + '_ohe')
    return df, ohe_cols


def get_attribute_ratio(df, numeric_cols, binary_cols, label_col):
    ratio_dict = {}
    if numeric_cols:
        avg_dict = df.select([sql.avg(c).alias(c) for c in numeric_cols]).first().asDict()
        ratio_num = (
            df.groupBy(label_col)
            .avg(*numeric_cols)
            .select([sql.max(col('avg(' + c + ')/{}'.format(avg_dict[c]))).alias(c) for c in numeric_cols])
            .fillna(0.0)
            .first()
            .asDict()
        )
        ratio_dict.update(ratio_num)
    if binary_cols:
        ratio_bin = (
            df.groupBy(label_col)
            .agg(*[(sql.sum(col(c)) / (sql.count(col(c)) - sql.sum(col(c)))).alias(c) for c in binary_cols])
            .fillna(1000.0)
            .select(*[sql.max(col(c)).alias(c) for c in binary_cols])
            .first()
            .asDict()
        )
        ratio_dict.update(ratio_bin)
    return OrderedDict(sorted(ratio_dict.items(), key=lambda v: -v[1]))


def select_features_by_ar(ar_dict, min_ar):
    return [f for f in ar_dict.keys() if ar_dict[f] >= min_ar]


def get_cluster_crosstab(df, cluster_col='cluster'):
    return (
        df.crosstab(cluster_col, 'labels2')
        .withColumn('count', col('attack') + col('normal'))
        .withColumn(cluster_col + '_labels2', col(cluster_col + '_labels2').cast('int'))
        .sort(col(cluster_col + '_labels2').asc())
    )


def split_clusters(crosstab):
    exp = ((col('count') > 25) & (col('attack') > 0) & (col('normal') > 0))
    cluster_rf = (
        crosstab.filter(exp).rdd
        .map(lambda row: (int(row['cluster_labels2']), [row['count'], row['attack'] / row['count']]))
        .collectAsMap()
    )
    cluster_mapping = (
        crosstab.filter(~exp).rdd
        .map(lambda row: (int(row['cluster_labels2']), 1.0 if (row['count'] <= 25) or (row['normal'] == 0) else 0.0))
        .collectAsMap()
    )
    return cluster_rf, cluster_mapping


def get_cluster_models(df, cluster_rf, ar_dict, labels2_indexer):
    cluster_models = {}
    labels_col = 'labels2_cl_index'
    labels2_indexer.setOutputCol(labels_col)
    rf_slicer = VectorSlicer(
        inputCol="indexed_features",
        outputCol="rf_features",
        names=select_features_by_ar(ar_dict, AR_THRESHOLD_FOR_RF),
    )
    for cluster in cluster_rf.keys():
        rf_classifier = RandomForestClassifier(
            labelCol=labels_col,
            featuresCol='rf_features',
            seed=SEED,
            numTrees=500,
            maxDepth=20,
            featureSubsetStrategy="sqrt",
        )
        rf_pipeline = Pipeline(stages=[labels2_indexer, rf_slicer, rf_classifier])
        cluster_models[cluster] = rf_pipeline.fit(df.filter(col('cluster') == cluster))
    return cluster_models


def confusion_metrics(preds, labels):
    cm = np.zeros((2, 2), dtype=int)
    for p, l in zip(preds, labels):
        cm[int(l)][int(p)] += 1
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    far = fp / max(1, (tn + fp))
    dr = tp / max(1, (tp + fn))
    precision = tp / max(1, (tp + fp))
    recall = dr
    f1 = 2 * precision * recall / max(1e-12, (precision + recall))
    return cm.tolist(), acc, far, dr, f1


class KMeansRFDetector:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.sc, self.sql_ctx = build_spark()
        self._is_trained = False

    def train(self):
        t0 = time()
        train_df = load_dataset(self.sc, self.train_path)
        test_df = load_dataset(self.sc, self.test_path)

        # Fix and drop
        train_df = train_df.replace(2.0, 0.0, 'su_attempted')
        test_df = test_df.replace(2.0, 0.0, 'su_attempted')
        train_df = train_df.drop('num_outbound_cmds')
        test_df = test_df.drop('num_outbound_cmds')
        numeric_cols = [c for c in NUMERIC_COLS if c != 'num_outbound_cmds']
        binary_cols = list(BINARY_COLS)

        # Labels
        labels2_indexer = StringIndexer(inputCol="labels2", outputCol="labels2_index")
        labels5_indexer = StringIndexer(inputCol="labels5", outputCol="labels5_index")
        labels_pipeline = Pipeline(stages=[Labels2Converter(), Labels5Converter(), labels2_indexer, labels5_indexer])
        train_df = labels_pipeline.fit(train_df).transform(train_df).withColumn('id', sql.monotonically_increasing_id()).cache()
        test_df = labels_pipeline.fit(test_df).transform(test_df).withColumn('id', sql.monotonically_increasing_id()).cache()

        # OHE
        train_ohe_cols = []
        train_df, ohe0 = ohe(train_df, NOMINAL_COLS[0]); train_ohe_cols += ohe0
        train_df, ohe1 = ohe(train_df, NOMINAL_COLS[1]); train_ohe_cols += ohe1
        train_df, ohe2 = ohe(train_df, NOMINAL_COLS[2]); train_ohe_cols += ohe2
        binary_cols_extended_train = binary_cols + train_ohe_cols

        test_ohe_cols = []
        test_df, tohe0 = ohe(test_df, NOMINAL_COLS[0]); test_ohe_cols += tohe0
        test_df, tohe1 = ohe(test_df, NOMINAL_COLS[1]); test_ohe_cols += tohe1
        test_df, tohe2 = ohe(test_df, NOMINAL_COLS[2]); test_ohe_cols += tohe2
        binary_cols_extended_test = binary_cols + test_ohe_cols

        # AR
        ar_dict = get_attribute_ratio(train_df, numeric_cols, binary_cols_extended_train, 'labels5')

        # Standardize
        avg_dict = train_df.select([sql.avg(c).alias(c) for c in numeric_cols]).first().asDict()
        std_dict = train_df.select([sql.stddev(c).alias(c) for c in numeric_cols]).first().asDict()

        def standardizer(column):
            return ((col(column) - avg_dict[column]) / std_dict[column]).alias(column)

        train_scaler_cols = [*binary_cols_extended_train, *[standardizer(c) for c in numeric_cols], *['id', 'labels2_index', 'labels2', 'labels5_index', 'labels5']]
        test_scaler_cols = [*binary_cols_extended_test, *[standardizer(c) for c in numeric_cols], *['id', 'labels2_index', 'labels2', 'labels5_index', 'labels5']]
        scaled_train_df = train_df.select(train_scaler_cols).cache()
        scaled_test_df = test_df.select(test_scaler_cols).cache()

        # Assemble and index
        assembler = VectorAssembler(inputCols=select_features_by_ar(ar_dict, AR_THRESHOLD_FOR_ASSEMBLER), outputCol='raw_features')
        indexer = VectorIndexer(inputCol='raw_features', outputCol='indexed_features', maxCategories=2)
        prep_model = Pipeline(stages=[assembler, indexer]).fit(scaled_train_df)
        scaled_train_df = prep_model.transform(scaled_train_df).select('id', 'indexed_features', 'labels2_index', 'labels2').cache()
        scaled_test_df = prep_model.transform(scaled_test_df).select('id', 'indexed_features', 'labels2_index', 'labels2').cache()

        # KMeans
        kmeans_slicer = VectorSlicer(
            inputCol='indexed_features', outputCol='features',
            names=list(set(select_features_by_ar(ar_dict, AR_THRESHOLD_FOR_KMEANS_NUMERIC)).intersection(numeric_cols)),
        )
        kmeans = KMeans(k=KMEANS_K, initSteps=KMEANS_INIT_STEPS, maxIter=KMEANS_MAX_ITER, featuresCol='features', predictionCol='cluster', seed=SEED)
        kmeans_model = Pipeline(stages=[kmeans_slicer, kmeans]).fit(scaled_train_df)

        kmeans_train_df = kmeans_model.transform(scaled_train_df).cache()
        kmeans_test_df = kmeans_model.transform(scaled_test_df).cache()

        # Per-cluster RF
        crosstab = get_cluster_crosstab(kmeans_train_df).cache()
        cluster_rf, cluster_mapping = split_clusters(crosstab)
        cluster_models = get_cluster_models(kmeans_train_df, cluster_rf, ar_dict, labels2_indexer)

        # Save members
        self.numeric_cols = numeric_cols
        self.binary_cols_extended_train = binary_cols_extended_train
        self.binary_cols_extended_test = binary_cols_extended_test
        self.avg_dict = avg_dict
        self.std_dict = std_dict
        self.prep_model = prep_model
        self.kmeans_model = kmeans_model
        self.cluster_mapping = cluster_mapping
        self.cluster_models = cluster_models
        self.labels2_indexer = labels2_indexer
        self.ar_dict = ar_dict

        # Build test result df
        kmeans_prob_col = 'kmeans_rf_prob'
        res_test_df = scaled_test_df.select('id', 'labels2_index', 'labels2').join(
            self.get_probabilities(kmeans_test_df, kmeans_prob_col), 'id'
        ).cache()

        preds_labels = res_test_df.select(kmeans_prob_col, 'labels2_index').rdd.map(lambda r: (1.0 if float(r[0]) >= THRESHOLD_TEST else 0.0, float(r[1]))).collect()
        preds = [p for p, _ in preds_labels]
        labels = [l for _, l in preds_labels]
        cm, acc, far, dr, f1 = confusion_metrics(preds, labels)

        self.metrics = {
            'confusion_matrix': cm,
            'accuracy': acc,
            'false_alarm_rate': far,
            'detection_rate': dr,
            'f1': f1,
            'threshold': THRESHOLD_TEST,
        }
        self._is_trained = True
        return {
            'train_time_sec': round(time() - t0, 2),
            **self.metrics,
        }

    def _standardizer(self, column):
        return ((col(column) - self.avg_dict[column]) / self.std_dict[column]).alias(column)

    def _transform_single(self, record_dict):
        # Create single-row DataFrame with all required columns
        # Fill missing numeric/binary columns with zeros
        data = {c: None for c in COL_NAMES.tolist()}
        data.update(record_dict)
        if 'labels' not in data:
            data['labels'] = 'normal'
        df = self.sql_ctx.createDataFrame([data])
        # Cast types
        for c in df.columns:
            if c in ['protocol_type', 'service', 'flag', 'labels']:
                df = df.withColumn(c, col(c).cast(StringType()))
            else:
                df = df.withColumn(c, col(c).cast(DoubleType()))
        # Fix and drop
        df = df.replace(2.0, 0.0, 'su_attempted')
        df = df.drop('num_outbound_cmds')
        # Labels (only to align pipeline expectations)
        df = Labels2Converter().transform(df)
        # OHE nominal on-the-fly using training side known categories is non-trivial with custom OHE.
        # For simplicity, rely on StringIndexer/VectorAssembler already embedded in prep_model,
        # which expects numeric and binary columns already present. We'll rebuild minimal columns:
        # Build binary extended by recomputing OHE for this single row
        df, _ = ohe(df, NOMINAL_COLS[0])
        df, _ = ohe(df, NOMINAL_COLS[1])
        df, _ = ohe(df, NOMINAL_COLS[2])
        # Standardize
        numeric_cols = [c for c in self.numeric_cols]
        scaler_cols = [*self.binary_cols_extended_test, *[self._standardizer(c) for c in numeric_cols]]
        df = df.select(*[c for c in scaler_cols if c in df.columns]).cache()
        # Add placeholders
        df = df.withColumn('id', sql.monotonically_increasing_id())
        df = df.withColumn('labels2_index', sql.when(col('labels2') == 'attack', sql.lit(1.0)).otherwise(sql.lit(0.0)))
        df = df.withColumn('labels2', col('labels2'))
        # prep -> features
        df = self.prep_model.transform(df).select('id', 'indexed_features', 'labels2_index', 'labels2')
        # kmeans -> cluster
        df = self.kmeans_model.transform(df)
        return df

    def get_probabilities(self, df, prob_col):
        pred_df = self.sql_ctx.createDataFrame([], StructType([
            StructField('id', LongType(), False),
            StructField(prob_col, DoubleType(), False),
        ]))
        udf_map = udf(lambda cluster: float(self.cluster_mapping[cluster]), DoubleType())
        keys = list(self.cluster_mapping.keys())
        if keys:
            mapped = (
                df.filter(col('cluster').isin(keys))
                  .withColumn(prob_col, udf_map(col('cluster')))
                  .select('id', prob_col)
            )
            pred_df = pred_df.union(mapped)
        for k in self.cluster_models.keys():
            maj_label = self.cluster_models[k].stages[0].labels[0]
            remap = udf(lambda row: float(row[0]) if (maj_label == 'attack') else float(row[1]), DoubleType())
            part = (
                self.cluster_models[k]
                .transform(df.filter(col('cluster') == k))
                .withColumn(prob_col, remap(col('probability')))
                .select('id', prob_col)
            )
            pred_df = pred_df.union(part)
        return pred_df

    def predict_one(self, record_dict):
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call /train first.")
        df = self._transform_single(record_dict)
        prob_col = 'kmeans_rf_prob'
        probs_df = self.get_probabilities(df, prob_col)
        row = probs_df.collect()[0]
        prob_attack = float(row[prob_col])
        pred = 1.0 if prob_attack >= THRESHOLD_TEST else 0.0
        return {
            'prob_attack': prob_attack,
            'prediction_index': pred,
            'prediction_label': LABELS2[int(pred)],
            'threshold': THRESHOLD_TEST,
        }

    def get_metrics(self):
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call /train first.")
        return self.metrics 