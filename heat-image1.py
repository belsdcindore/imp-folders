# import tkinter as tk
# from tkinter import filedialog, messagebox
# from ultralytics import YOLO
# import cv2
# from PIL import Image, ImageTk

# # ---------------------------
# # Load YOLOv8 model directly online
# # ---------------------------
# def load_model():
#     global model
#     try:
#         # Option 1: Ultralytics official pretrained (automatic download)
#         model = YOLO("yolov8s.pt")  

#         # Option 2: Your own model hosted online (HuggingFace / GitHub)
#         # model = YOLO("https://huggingface.co/username/repo-name/resolve/main/best.pt")

#         messagebox.showinfo("Success", "Model loaded successfully from online!")
#     except Exception as e:
#         messagebox.showerror("Error", f"Failed to load model: {e}")

# # ---------------------------
# # Select image and run inference
# # ---------------------------
# def crowd_management_from_image():
#     try:
#         if 'model' not in globals():
#             messagebox.showerror("Error", "Please load the model first!")
#             return

#         # Select image
#         file_path = filedialog.askopenfilename(
#             filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
#         )
#         if not file_path:
#             return

#         # Run YOLO inference
#         results = model(file_path)

#         # Save output image with detections
#         results[0].save(filename="output.jpg")

#         # Show in GUI
#         img = Image.open("output.jpg")
#         img = img.resize((600, 400))
#         img_tk = ImageTk.PhotoImage(img)
#         panel.config(image=img_tk)
#         panel.image = img_tk

#         # Show total detected people
#         count = len(results[0].boxes)
#         messagebox.showinfo("Crowd Count", f"Estimated People Detected: {count}")

#     except Exception as e:
#         messagebox.showerror("Error", f"Failed to process image: {e}")

# # ---------------------------
# # GUI Setup
# # ---------------------------
# root = tk.Tk()
# root.title("Crowd Management using YOLOv8 (Online Model)")
# root.geometry("800x600")
# root.configure(bg="lightblue")

# # Buttons
# btn_load = tk.Button(root, text="Generate & Load YOLOv8 Model", command=load_model, width=40)
# btn_load.pack(pady=10)

# btn_infer = tk.Button(root, text="Crowd Management from Images", command=crowd_management_from_image, width=40)
# btn_infer.pack(pady=10)

# btn_exit = tk.Button(root, text="Exit", command=root.destroy, width=20)
# btn_exit.pack(pady=10)

# # Panel for displaying image
# panel = tk.Label(root)
# panel.pack(pady=20)

# root.mainloop()


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# CSRNet Model (simplified)
# ---------------------------
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512]
        self.backend_feat = [512,512,512,256,128,64]
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat,in_channels=512,dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        layers = []
        d_rate = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# ---------------------------
# Load pretrained CSRNet weights (downloaded separately)
# ---------------------------
model = CSRNet()
# NOTE: you need pretrained weights file "csrnet_pretrained.pth"
# e.g. from: https://github.com/leeyeehoo/CSRNet-pytorch
# model.load_state_dict(torch.load("csrnet_pretrained.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# ---------------------------
# Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Load your image
img_path = "1_5.jpg"  # replace with your image
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# ---------------------------
# Inference
# ---------------------------
with torch.no_grad():
    density_map = model(img_tensor).cpu().numpy()[0,0]

crowd_count = int(density_map.sum())

print(f"Estimated Crowd Count: {crowd_count}")

# ---------------------------
# Show Density Map
# ---------------------------
plt.imshow(density_map, cmap="jet")
plt.title(f"Crowd Count: {crowd_count}")
plt.axis("off")
plt.show()
