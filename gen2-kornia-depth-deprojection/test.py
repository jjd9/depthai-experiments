import struct
def float16_to_uint8(val):
    return list(bytearray(struct.pack("e", val)))
while True:
    img = node.io['in'].get()
    img.setType(ImgFrame.Type.GRAYF16)
    node.warn("img type: {}".format(img.getType()))
    new_buffer = []
    old_buffer = img.getData()
    for i in range(0, old_buffer, 2):
        val_uint16 = 256 * old_buffer[1] + old_buffer[0]
        new_buffer += float16_to_uint8(val_uint16)
    img.setData(new_buffer)
    node.io['out'].send(img)
