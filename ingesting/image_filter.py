from PIL import Image
import os

# --- 过滤器可调参数 ---
# 你可以在这里调整过滤器的灵敏度

# 1. 尺寸过滤器
# 图片面积占页面面积的最小/最大比例，超过这个范围的将被过滤
MIN_IMAGE_AREA_RATIO = 0.02  # 过滤掉小于页面面积 2% 的小图 (logo, icon)
MAX_IMAGE_AREA_RATIO = 0.80  # 过滤掉大于页面面积 80% 的大图 (背景)

# 2. 颜色过滤器
# 图片中的颜色种类如果小于这个数值，将被认为是纯色图或简单图
MIN_UNIQUE_COLORS = 15  # 如果颜色种类少于15种，很可能是简单背景或废图


def is_image_valuable(image_path, page_area):
    """
    判断一张图片是否“有价值”，综合尺寸和颜色复杂度进行判断。

    参数:
    - image_path (str): 需要检查的图片文件路径。
    - page_area (float): 图片所在PDF页面的总面积，用于尺寸比较。

    返回:
    - (bool, str): 一个元组，第一个元素为 True (有价值) 或 False (无价值)，
                   第二个元素为判断的理由。
    """
    try:
        # --- 1. 按尺寸过滤 ---
        img_size = os.path.getsize(image_path)
        if img_size == 0:
            return False, f"过滤-文件为空: {os.path.basename(image_path)}"

        with Image.open(image_path) as pil_img:
            img_width, img_height = pil_img.size
            img_area = img_width * img_height

            if page_area > 0:
                area_ratio = img_area / page_area
                if area_ratio < MIN_IMAGE_AREA_RATIO:
                    return False, f"过滤-尺寸过小: {os.path.basename(image_path)} (面积占比 {area_ratio:.2%})"
                if area_ratio > MAX_IMAGE_AREA_RATIO:
                    return False, f"过滤-尺寸过大: {os.path.basename(image_path)} (面积占比 {area_ratio:.2%})"

            # --- 2. 按颜色过滤 ---
            # getcolors() 对于颜色过多的图片会返回 None，这正是我们想要的复杂图片
            # 我们设置一个较大的 maxcolors 值来检查
            colors = pil_img.getcolors(maxcolors=img_width * img_height)
            if colors is not None and len(colors) < MIN_UNIQUE_COLORS:
                return False, f"过滤-颜色单一: {os.path.basename(image_path)} (颜色数 {len(colors)})"

    except Exception as e:
        # 如果图片处理出错（例如格式损坏），也认为它没有价值
        return False, f"过滤-处理异常: {os.path.basename(image_path)} ({e})"

    # 如果通过所有检查，则认为是有价值的
    return True, "有价值的图表"