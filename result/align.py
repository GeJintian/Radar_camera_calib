import json
import os


def matching(rgb_sec,rgb_nse,i,radar_files):
    val = int(rgb_sec)*1000+int(rgb_nse)
    #print(val)
    rad_sec = radar_files[i].split('_')[0][-3:]
    rad_nse = radar_files[i].split('_')[1][:3]
    key = int(rad_sec)*1000+int(rad_nse)
    if key >= val:
        return abs(val-key), i, radar_files[i]

    while key < val:
        key_pre = key
        if i == len(radar_files)-1:
            return abs(key_pre-val), i, radar_files[i]
        i = i + 1

        rad_sec = radar_files[i].split('_')[0][-3:]
        rad_nse = radar_files[i].split('_')[1][:3]
        key = int(rad_sec)*1000+int(rad_nse)
    err0 = abs(key_pre - val)
    err1 = abs(key-val)
    if err0 > err1:
        return err1, i, radar_files[i]
    else:
        i = i - 1
        return err0, i, radar_files[i]
    

if __name__ == '__main__':
    rgb_path = 'img'
    radar_path = 'radar'

    rgb_files = sorted(os.listdir(rgb_path))
    radar_files = sorted(os.listdir(radar_path))

    alignment = {}
    i = 0
    for rgb in rgb_files:
        rgb_sec = rgb.split('_')[0][-3:]
        rgb_nse = rgb.split('_')[1][:3]
        _,i,rad = matching(rgb_sec, rgb_nse, i,radar_files)
        i = i + 1
        alignment[rgb] = rad
    #print(alignment)
    b = json.dumps(alignment)
    f = open('alignment.json','w')
    f.write(b)
    f.close()