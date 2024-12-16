import base64
import cv2
import numpy as np

def steg_encode(input_image, key):
    # Basic LSB for now
    contents = input_image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert key to binary
    key_bin = ''.join(format(ord(i), '08b') for i in key)
    
    # encode lsb
    for i in range(len(key_bin)):
        img[i//img.shape[1], i%img.shape[1], 0] = (img[i//img.shape[1], i%img.shape[1], 0] & 0b11111110) | int(key_bin[i])

    _, encoded_img = cv2.imencode('.PNG', img)
    return base64.b64encode(encoded_img)
    

def steg_decode(img):
    # Basic LSB for now
    contents = img.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # decode lsb
    key_bin = ''
    for i in range(192*3):
        key_bin += str(img[i//img.shape[1], i%img.shape[1], 0] & 1)

    key = ''
    for i in range(0, len(key_bin), 8):
        key += chr(int(key_bin[i:i+8], 2))

    return key
