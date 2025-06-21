import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from DataProcess import clean_text

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils import resample
from scipy import sparse

def load_and_prepare_data(csv_path):
    """读取并清洗数据"""
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.columns = df.columns.str.strip()
    df['cleaned_text'] = df['text'].apply(clean_text)
    X = df['cleaned_text']
    y = df['label'].map({'ham': 0, 'spam': 1})
    return X, y

def balance_data(X_vec, y):
    # 将 minority class spam 上采样
    X_df = pd.DataFrame(X_vec.toarray())  # 稀疏矩阵转 DataFrame
    y_df = pd.Series(y)

    df_all = pd.concat([X_df, y_df.rename("label")], axis=1)

    spam_df = df_all[df_all['label'] == 1]
    ham_df = df_all[df_all['label'] == 0]

    spam_upsampled = resample(spam_df, replace=True, n_samples=len(ham_df), random_state=42)

    df_balanced = pd.concat([ham_df, spam_upsampled])
    X_balanced = df_balanced.drop(columns=['label']).values
    y_balanced = df_balanced['label'].values

    return X_balanced, y_balanced


def vectorize_text(X):
    """TF-IDF 向量化"""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # 添加 bigrams 提升上下文建模
        min_df=2,            # 忽略仅出现一次的词
        max_df=0.95,         # 忽略过于频繁的词
        sublinear_tf=True    # 缩放 tf 值（对 SVM 更有效）
    )
    X_vec = vectorizer.fit_transform(X)
    return X_vec, vectorizer


def train_ensemble_with_cv():
    print("📂 Loading data...")
    X, y = load_and_prepare_data(r"/Users/fairchan/Desktop/人工智能原理/python/spam_dataset.csv")
    X_vec, vectorizer = vectorize_text(X)

    # VotingClassifier（support soft voting）
    ensemble = VotingClassifier(
        estimators=[
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('nb', MultinomialNB()),
            ('svm', SVC(probability=True, class_weight='balanced', kernel='linear'))  # Use linear kernels to speed up
        ],
        voting='soft',
        # weights=[1, 2, 3]
    )

    # === Add K-fold cross validation assessment ===
    print("🔎 5-fold cross validation in progress...")
    cv_scores = cross_val_score(ensemble, X_vec, y, cv=5, scoring='accuracy')
    print(f"✅ 5-fold cross validation accuracy: {cv_scores}")
    print(f"📊 Average accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ✅ === Start Practical Training and Preservation ===
    print("\n🚀 Training...")
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # 1. Divide the data (don't break it up yet)
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # 2. Up-sample the training set (X_vec is a sparse matrix)
    from scipy.sparse import vstack
    from sklearn.utils import resample

    # Converting the training set into a DataFrame for easy sampling
    X_train_df = pd.DataFrame(X_train_raw.toarray())
    y_train_df = pd.Series(y_train_raw).reset_index(drop=True)
    df_train = pd.concat([X_train_df, y_train_df.rename("label")], axis=1)

    # Splitting ham and spam
    ham = df_train[df_train.label == 0]
    spam = df_train[df_train.label == 1]

    # Up-sample spam so that number = ham
    spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)

    # Consolidation of data
    df_balanced = pd.concat([ham, spam_upsampled])
    X_bal = df_balanced.drop(columns=['label']).values
    y_bal = df_balanced['label'].values
    
    X_bal_sparse = sparse.csr_matrix(X_bal)
    # 3. Training the model with up-sampling results

    ensemble.fit(X_bal_sparse, y_bal)

    y_pred = ensemble.predict(X_test)
    print("\n📈 Final Test Set Performance.")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    joblib.dump({'model': ensemble, 'vectorizer': vectorizer}, "ensemble_spam_classifier.pkl")
    print("\n✔ The fusion model has been saved as:ensemble_spam_classifier.pkl")

    # 4. Training models
    print("🚀 ensemble model being trained... Model evaluation results:")
    ensemble.fit(X_train, y_train)

    # 5. Model evaluation
    print("\n📊 Model evaluation results:")
    y_pred = ensemble.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    # 6. Preservation of models (containing models + vectors)
    combined = {
        "model": ensemble,
        "vectorizer": vectorizer
    }
    joblib.dump(combined, "ensemble_spam_classifier.pkl")
    print("\n✅ Theensemble model seved as : ensemble_spam_classifier.pkl")

    return ensemble


if __name__ == "__main__":
    train_ensemble_with_cv()