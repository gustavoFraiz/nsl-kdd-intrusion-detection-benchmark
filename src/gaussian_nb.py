# Gaussian Naive Bayes experiment recovered from the NSL-KDD benchmark.

import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from memory_profiler import memory_usage

start_time = time.time()
start_mem = memory_usage()[0]

train_path = 'KDDTrain+.txt'
test_path = 'KDDTest+.txt'

column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

attack_mapping = {
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos',
    'udpstorm': 'dos', 'processtable': 'dos', 'mailbomb': 'dos', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe', 'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l',
    'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    'httptunnel': 'r2l', 'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
}

df_train = pd.read_csv(train_path, header=None, names=column_names)
df_test = pd.read_csv(test_path, header=None, names=column_names)
df_train.drop(columns=['difficulty'], inplace=True)
df_test.drop(columns=['difficulty'], inplace=True)
df_train['label'] = df_train['label'].map(attack_mapping).fillna(df_train['label'])
df_test['label'] = df_test['label'].map(attack_mapping).fillna(df_test['label'])

X_train_raw = df_train.drop('label', axis=1)
y_train_raw = df_train['label']
X_test_raw = df_test.drop('label', axis=1)
y_test_raw = df_test['label']

categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = X_train_raw.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

model = GaussianNB()
model.fit(X_train_resampled, y_train_resampled)

y_pred_int = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)
y_pred_str = label_encoder.inverse_transform(y_pred_int)
y_test_str = label_encoder.inverse_transform(y_test)
all_class_names = list(label_encoder.classes_)

cm = confusion_matrix(y_test_str, y_pred_str, labels=all_class_names)
cm_df = pd.DataFrame(cm, index=all_class_names, columns=all_class_names)
plt.figure(figsize=(12, 9))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(y_test_str, y_pred_str, labels=all_class_names, digits=4, zero_division=0))
auc_roc_score_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
print(f'Weighted OvR AUC-ROC: {auc_roc_score_val:.4f}')

fp_mask = (np.array(y_test_str) == 'normal') & (np.array(y_pred_str) != 'normal')
fn_mask = (np.array(y_test_str) != 'normal') & (np.array(y_pred_str) == 'normal')
print('False positives:', int(fp_mask.sum()))
print('False negatives:', int(fn_mask.sum()))

end_time = time.time()
end_mem = memory_usage()[0]
print(f'Total time: {end_time - start_time:.2f}s')
print(f'Memory used: {end_mem - start_mem:.2f} MB')
