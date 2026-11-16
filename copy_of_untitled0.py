import os
import csv
import json
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

all_codes = []
all_labels = []

data_files = [
    "cvefixes_clean (1).csv",
    "dataset.jsonl",
    "gh_top_1000_projects_vulns.jsonl"
]

for fname in data_files:
    if not os.path.exists(fname):
        print(f"الملف غير موجود: {fname}")
        continue
    print(f"جاري قراءة الملف: {fname}")

    if fname.endswith('.csv'):
        with open(fname, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    code_val = next((v for v in row.values() if v and len(str(v)) > 15), None)
                    label_val = next((v for v in row.values() if str(v).strip().lower() in ["0", "1", "true", "false", "secure", "vulnerable", "buggy", "insecure"]), None)
                    if code_val:
                        all_codes.append(str(code_val))
                        lbl = 1 if label_val and str(label_val).lower() in ["1", "true", "vulnerable", "buggy", "insecure"] else 0
                        all_labels.append(lbl)
                except:
                    continue

    elif fname.endswith(('.json', '.jsonl')):
        with open(fname, "r", encoding="utf-8") as f:
            try:
                first_char = f.read(1)
                f.seek(0)
                items = json.load(f) if first_char == '[' else [json.loads(line) for line in f if line.strip()]

                for item in items:
                    if isinstance(item, dict):
                        text_fields = {k: str(v) for k, v in item.items() if isinstance(v, str) and len(v) > 15}
                        if text_fields:
                            best_code_key = max(text_fields, key=lambda k: len(text_fields[k]))
                            code_val = text_fields[best_code_key]

                            lbl = None
                            for k, v in item.items():
                                val_str = str(v).strip().lower()
                                if val_str in ["1", "true", "vulnerable", "buggy", "insecure"] or (isinstance(v, int) and v > 0):
                                    lbl = 1
                                    break
                                elif val_str in ["0", "false", "secure", "safe"]:
                                    lbl = 0
                                    break

                            if lbl is None:
                                lbl = 1 if ("vuln" in fname.lower() or "bug" in fname.lower()) else 0

                            all_codes.append(code_val)
                            all_labels.append(lbl)
            except Exception as e:
                print(f"خطأ في قراءة الملف {fname}: {e}")

print(f"إجمالي الدوال المجمعة: {len(all_codes)}")

Xtr, Xte, ytr, yte = train_test_split(
    all_codes, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

vec = TfidfVectorizer(max_features=20000)
Xtr_v = vec.fit_transform(Xtr)

pos_count = ytr.count(1)
neg_count = len(ytr) - pos_count
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

model = XGBClassifier(n_estimators=200, scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)
model.fit(Xtr_v, ytr)
print("تم تدريب النموذج بنجاح!")

def predict_vulnerability(code_snippet):
    if not code_snippet.strip():
        return "الرجاء لصق كود أولاً."
    vectorized_input = vec.transform([code_snippet])
    prediction = model.predict(vectorized_input)[0]

    if prediction == 1:
        return "⚠️ تحذير: الكود مصاب أو يحتوي على ثغرة (Vulnerable)"
    else:
        return "✅ الكود آمن (Secure)"

demo = gr.Interface(
    fn=predict_vulnerability,
    inputs=gr.Textbox(lines=10, placeholder="الصق كود البرمجة هنا..."),
    outputs="text",
    title="فاحص ثغرات الأكواد بالذكاء الاصطناعي",
    description="قم بلصق أي دالة أو كود برمجي لمعرفة ما إذا كان آمناً أم يحتوي على ثغرات أمنية."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
