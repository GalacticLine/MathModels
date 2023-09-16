""" 取色器测试 """
import os.path
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_sample_image
from Plot.colors import ColorGetter

path = 'china.jpg'

# 加载并保存图片
if not os.path.exists(path):
    data = load_sample_image(path)
    image = Image.fromarray(data)
    image.save(path)
    image.show()

# 从图片中获取指定数量颜色
getter = ColorGetter.from_image(path, 10)
print(getter.get_hex_data())
getter.plot()
plt.show()


