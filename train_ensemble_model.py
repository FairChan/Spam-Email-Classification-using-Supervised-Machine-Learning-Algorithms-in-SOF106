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
    print("📂 正在加载数据...")
    X, y = load_and_prepare_data(r"/Users/fairchan/Desktop/人工智能原理/python/spam_dataset.csv")
    X_vec, vectorizer = vectorize_text(X)

    # 构建 VotingClassifier（支持 soft voting）
    ensemble = VotingClassifier(
        estimators=[
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('nb', MultinomialNB()),
            ('svm', SVC(probability=True, class_weight='balanced', kernel='linear'))  # 使用线性核以加速
        ],
        voting='soft',
        weights=[1, 2, 3]
    )

    # ✅ === 加入 K 折交叉验证评估 ===
    print("🔎 正在进行 5 折交叉验证...")
    cv_scores = cross_val_score(ensemble, X_vec, y, cv=5, scoring='accuracy')
    print(f"✅ 5-fold 交叉验证准确率: {cv_scores}")
    print(f"📊 平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ✅ === 开始实际训练与保存 ===
    print("\n🚀 正式训练融合模型...")
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # 1. 划分数据（先不打乱）
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # 2. 对训练集进行上采样（X_vec 是稀疏矩阵）
    from scipy.sparse import vstack
    from sklearn.utils import resample

    # 将训练集转成 DataFrame，便于采样
    X_train_df = pd.DataFrame(X_train_raw.toarray())
    y_train_df = pd.Series(y_train_raw).reset_index(drop=True)
    df_train = pd.concat([X_train_df, y_train_df.rename("label")], axis=1)

    # 拆分 ham 和 spam
    ham = df_train[df_train.label == 0]
    spam = df_train[df_train.label == 1]

    # 对 spam 进行上采样，使其数量 = ham
    spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)

    # 合并数据
    df_balanced = pd.concat([ham, spam_upsampled])
    X_bal = df_balanced.drop(columns=['label']).values
    y_bal = df_balanced['label'].values
    
    X_bal_sparse = sparse.csr_matrix(X_bal)
    # 3. 拿上采样结果训练模型

    ensemble.fit(X_bal_sparse, y_bal)

    y_pred = ensemble.predict(X_test)
    print("\n📈 最终测试集表现:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    joblib.dump({'model': ensemble, 'vectorizer': vectorizer}, "ensemble_spam_classifier.pkl")
    print("\n✔ 融合模型已保存为：ensemble_spam_classifier.pkl")

    # 4. 训练模型
    print("🚀 正在训练融合模型...")
    ensemble.fit(X_train, y_train)

    # 5. 模型评估
    print("\n📊 模型评估结果：")
    y_pred = ensemble.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    # 6. 保存模型（包含模型 + 向量器）
    combined = {
        "model": ensemble,
        "vectorizer": vectorizer
    }
    joblib.dump(combined, "ensemble_spam_classifier.pkl")
    print("\n✅ 融合模型保存为：ensemble_spam_classifier.pkl")

    return ensemble


if __name__ == "__main__":
    train_ensemble_with_cv()