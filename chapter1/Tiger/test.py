from PIL import Image
import cv2


def main():
    # img = Image.open('/home/enningxie/Documents/DataSets/butterfly_/data_augmentation/AAaa0007008_1.jpg')
    # img.save('/home/enningxie/Documents/DataSets/butterfly_/data_augmentation/AAaa0007008_1.png')

    img = cv2.imread('/home/enningxie/Documents/DataSets/butterfly_/data_augmentation/AAaa0007008_1.jpg')
    # print(filename.replace(".jpg", ".png"))
    # newfilename = filename.replace(".jpg", ".png")
    # cv2.imshow("Image",img)
    # cv2.waitKey(0)
    cv2.imwrite('/home/enningxie/Documents/DataSets/butterfly_/data_augmentation/AAaa0007008_1_.png', img)

if __name__ == '__main__':
    main()