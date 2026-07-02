import torch
import torch.nn.functional as F
from define import MNISTNet
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# 加载模型
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(DEVICE)
model.load_state_dict(torch.load('models/mnist_net.pth', map_location=DEVICE))
model.eval()

def load_custom_image(image_path):
    """加载并预处理自定义图片"""
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    if img_array.mean() > 128:
        img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).float()
    return img_tensor

def predict_image(image_path):
    """预测图片中的数字"""
    image = load_custom_image(image_path)
    with torch.no_grad():
        output = model(image.to(DEVICE))
        probabilities = F.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted].item()
    return predicted, confidence, probabilities[0].tolist()

def select_image():
    """选择图片并进行预测"""
    file_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.webp;*.bmp")]
    )
    if not file_path:
        return
    try:
        img = Image.open(file_path)
        img = img.resize((150, 150), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_canvas.delete("all")
        image_canvas.create_image(75, 75, image=photo)
        image_canvas.photo = photo
        
        predicted, confidence, probabilities = predict_image(file_path)
        result_label.config(text=f"预测数字: {predicted}")
        confidence_label.config(text=f"置信度: {confidence:.2%}")
        
        for i in range(10):
            bars[i]['value'] = probabilities[i] * 100
            prob_labels[i].config(text=f"{probabilities[i]:.1%}")
        
        status_label.config(text=f"已处理: {file_path}", foreground="green")
    except Exception as e:
        messagebox.showerror("错误", f"处理图片时发生错误: {e}")
        status_label.config(text=f"错误: {e}", foreground="red")

# 创建主窗口
root = tk.Tk()
root.title("MNIST 手写数字识别")
root.geometry("600x700")
root.resizable(True, True)

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12))
style.configure('TLabel', font=('Helvetica', 12))

main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# 标题
title_label = ttk.Label(main_frame, text="手写数字识别系统", font=('Helvetica', 18, 'bold'))
title_label.pack(pady=(0, 20))

# 图片显示
image_frame = ttk.Frame(main_frame)
image_frame.pack(pady=10)
image_canvas = tk.Canvas(image_frame, width=150, height=150, borderwidth=2, relief="solid", bg="white")
image_canvas.pack()
image_canvas.create_text(75, 75, text="点击下方按钮选择图片", fill="gray", font=('Helvetica', 10))

# 选择按钮
select_btn = ttk.Button(main_frame, text="选择图片", command=select_image)
select_btn.pack(pady=10)

# 预测结果
result_frame = ttk.Frame(main_frame)
result_frame.pack(pady=10, fill=tk.X)
result_label = ttk.Label(result_frame, text="预测数字: -")
result_label.pack(side=tk.LEFT, padx=20)
confidence_label = ttk.Label(result_frame, text="置信度: -")
confidence_label.pack(side=tk.RIGHT, padx=20)

# 概率分布
prob_frame = ttk.Frame(main_frame)
prob_frame.pack(pady=10, fill=tk.X)
prob_title = ttk.Label(prob_frame, text="各数字概率分布:", font=('Helvetica', 11, 'bold'))
prob_title.pack(anchor=tk.W, pady=(0, 5))

bars = []
prob_labels = []
for i in range(10):
    row = ttk.Frame(prob_frame)
    row.pack(fill=tk.X, pady=2)
    num_label = ttk.Label(row, text=str(i), width=3)
    num_label.pack(side=tk.LEFT)
    bar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=400, mode='determinate')
    bar.pack(side=tk.LEFT, padx=5)
    bars.append(bar)
    prob_label = ttk.Label(row, text="0%", width=8)
    prob_label.pack(side=tk.LEFT)
    prob_labels.append(prob_label)

# 状态
status_label = ttk.Label(main_frame, text="就绪", foreground="gray")
status_label.pack(pady=(10, 0))

if __name__ == '__main__':
    root.mainloop()