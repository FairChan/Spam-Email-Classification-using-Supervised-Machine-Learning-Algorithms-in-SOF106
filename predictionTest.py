import joblib
import pandas as pd
from DataProcess import clean_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_model():
    model_data = joblib.load("ensemble_spam_classifier.pkl")
    return model_data['model'], model_data['vectorizer']


def predict_single(text, model, vectorizer):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    return 'spam' if pred == 1 else 'ham'

def evaluate_batch(csv_path, model, vectorizer):
    df = pd.read_csv(csv_path)
    df['cleaned'] = df['text'].apply(clean_text)
    X = vectorizer.transform(df['cleaned'])
    y_true = df['label'].str.strip().str.lower()
    y_pred = model.predict(X)


    label_map = {0: 'ham', 1: 'spam'}
    y_pred_labels = [label_map[i] for i in y_pred]

    print("\n=== 📋 每条邮件的判断结果 ===")
    for i in range(len(df)):
        print(f"\n📩 邮件内容：{df['text'].iloc[i]}")
        print(f"✅ 预测结果：{y_pred_labels[i].upper()}")
        print(f"🎯 实际标签：{y_true.iloc[i].upper()}")
        print(f"{'✔ 正确' if y_pred_labels[i] == y_true.iloc[i] else '❌ 错误'}")

    print("\n=== 📊 批量检测评估结果 ===")
    print("✅ 准确率:", accuracy_score(y_true, y_pred_labels))
    print("📋 分类报告:\n", classification_report(y_true, y_pred_labels, target_names=['ham', 'spam']))
    print("🔢 混淆矩阵:\n", confusion_matrix(y_true, y_pred_labels))


if __name__ == "__main__":
    model, vectorizer = load_model()

    print("请选择操作方式：")
    print("1 - 输入一条邮件进行判断")
    print("2 - 对一个CSV文件批量判断并评估准确率")
    choice = input("输入数字选择（1或2）：").strip()

    if choice == '1':
        email = input("\n📩 请输入一条邮件内容：\n>> ")
        result = predict_single(email, model, vectorizer)
        print(f"\n📤 预测结果：{result.upper()}")

    elif choice == '2':
        path = "/Users/fairchan/Desktop/人工智能原理/python/synthetic_spam_ham_dataset.csv"
        evaluate_batch(path, model, vectorizer)

    else:
        print("❌ 无效输入，请选择 1 或 2。")