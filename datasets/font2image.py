from ttf_utils import *
from tqdm import tqdm

char_list = open('D:\\cv_Dataset\\ETL9B-71\\char_list.txt', 'r', encoding='utf-8').read()

font_file = 'D:\\cv_Dataset\\kuzushiji-49\\font'
image_file = 'D:\\cv_Dataset\\ETL9B-71\\glyph'
fonts = os.listdir(font_file)

for font in tqdm(fonts):
    font_path = os.path.join(font_file, font)
    print(font_path)
    try:
        font2image(font_path, image_file, char_list, 128)
        print(font)
    except Exception as e:
        print(e)
remove_empty_floder(image_file)
