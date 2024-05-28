import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import os
import cv2

current_path = os.path.abspath(__file__)
path_to_image = r"Images\brute_images\mlfbList1.png"
img_path = os.path.join(os.path.dirname(current_path), path_to_image)

print("Path to image: ", img_path)
table_extractor = te.TableExtractor(img_path)
perspective_corrected_image = table_extractor.execute()
cv2.imshow("perspective_corrected_image", perspective_corrected_image)


lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
image_without_lines = lines_remover.execute()
cv2.imshow("image_without_lines", image_without_lines)

ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
ocr_tool.execute()

cv2.waitKey(0)
cv2.destroyAllWindows()