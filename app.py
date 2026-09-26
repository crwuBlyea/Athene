import gradio as gr
import numpy as np
from PIL import Image
import onnxruntime as ort
from pathlib import Path
import json
W = Path(__file__).parent / "weights"
CFG = json.load(open(W / "athene_config.json"))
MEAN = np.array(CFG["preprocessing"]["mean"], dtype=np.float32)
STD  = np.array(CFG["preprocessing"]["std"],  dtype=np.float32)
THR  = CFG["threshold_fpr002"]

sess_t = ort.InferenceSession(str(W / CFG["teacher"]),
                              providers=["CPUExecutionProvider"])
sess_s = ort.InferenceSession(str(W / CFG["student"]),
                              providers=["CPUExecutionProvider"])

def views(img: Image.Image):
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    img = img.resize((256, 256), Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)
    out = []
    for (x0, y0), flip in [((16,16), False), ((16,16), True),
                            ((0,16), False), ((32,16), False)]:
        v = x[:, y0:y0+224, x0:x0+224]
        out.append(v[:, :, ::-1] if flip else v)
    return out

def p1(logits):
    z = logits[1] - logits[0]
    return 1 / (1 + np.exp(-z))

def predict(img):
    if img is None:
        return None, "no image"
    img = img.convert("RGB")
    ps = []
    for v in views(img):
        lt = sess_t.run(None, {"input": v[None].astype(np.float32)})[0][0]
        ls = sess_s.run(None, {"input": v[None].astype(np.float32)})[0][0]
        ps.append(0.7 * p1(lt) + 0.3 * p1(ls))
    p = float(np.mean(ps))

    if p > 0.95 or p < 0.05:
        verdict, advice = ("AI-generated" if p > 0.5 else "Human-made"), "confident"
    else:
        verdict = "AI-generated" if p > THR else "Human-made"
        advice = "⚠ borderline — human review recommended"

    bar = f"P(AI) = {p:.3f}"
    label = {"AI-generated": p, "Human-made": 1 - p}
    return label, f"{bar}  ·  {advice}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="artwork"),
    outputs=[gr.Label(num_top_classes=2, label="verdict"),
             gr.Text(label="details")],
    title="🦉 Athene — AI-art detector",
    description=("Detects AI-generated digital art (anime, illustration, concept art). "
                 "Calibrated probabilities — they mean what they say."),
    article=("**Tested on:** Midjourney, Civitai/SD, Ideogram, DALL-E 3 · "
             "**Fails on:** classical painting scans · "
             "**Borderline results** mean exactly that: human review recommended. "
             "At strict settings ~half of unseen-generator AI is caught — "
             "that's the field's frontier, stated honestly."),
)

demo.launch(theme=gr.themes.Base())