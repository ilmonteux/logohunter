################################################################################
# This script was originally created by the authors of the Logos In The Wild
# dataset at Fraunhofer IOSB, and is licensed under CC-by-sa-4.0
# I have made only small modifications to the original script distributed with
# the dataset, as it was not running out of the box.
################################################################################
# -*- coding: iso-8859-1 -*-
from shutil import copy2
import shutil
import os
import xml.etree.ElementTree
import cv2
import argparse


ext = '.jpg'
dstext = '.jpg'
postfix = ''

fl32_intersection = ["adidas-text", "adidas1", "adidas2", "adidas3", "aldi", "aldi-text", "aldinord", "apple",
                     "becks", "becks-symbol", "bmw", "carlsberg", "coca-cola", "coke", "corona-text", "corona-symbol",
                     "dhl", "esso", "esso-text", "federalexpress", "fedex", "ferrari", "ford-symbol", "google-text",
                     "google+", "google-symbol", "guinness", "heineken", "hp", "milka", "nvidia", "paulaner",
                     "pepsi-text", "pepsi-symbol", "shell-symbol", "shell-text", "starbucks-text", "starbucks-symbol",
                     "stellaartois", "tsingtao", "ups"]

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Clean Logos In The Wild dataset')
    parser.add_argument('--in', dest='inpath', help='Path of the original dataset\'s data folder to be cleaned. It won\'t be modified.',
                        default=None, type=str, required=True)
    parser.add_argument('--out', dest='outpath',
                        help='Path where the cleaned dataset will be copied.',
                        default=None, type=str, required=True)
    parser.add_argument('--wofl32', dest='wofl32',
                    help='Generate the dataset without the classes of FlickrLogos32.',
                    action='store_true', default=False)
    parser.add_argument('--roi', dest='roi',
                    help='Writes the rois out for each brands separately.',
                    action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    imglist = ''
    brandlist = list()
    print('Copying dataset')
    if os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    xmlpath = os.path.join(args.outpath, 'voc_format')
    shutil.copytree(args.inpath, xmlpath)

    i = 0
    unavailableCounter = 0
    imageCounter = 0
    totalRoiCounter = 0
    print('Processing dataset files')
    for subdir_counter, (r, subdirs, files) in enumerate(os.walk(xmlpath)):
        subdirs.sort(key=str.casefold)
        for filename in files:
            i = i + 1
            if not filename.endswith('.xml'):
                continue
            if i % 1000 == 0:
                print('Processed: ' + str(i) + ' images.')
            imagename = filename.split('.')[0]
            imgpath = os.path.join(r, imagename + ext)
            filewithpath = os.path.join(r, filename)
            # test that image is actual image (sometimes 404 HTMLs downloaded as images)
            im = cv2.imread(os.path.join(r, imagename + ext))
            if (not os.path.isfile(imgpath)) or (im is None):
                os.remove(filewithpath)
                unavailableCounter += 1
                print('Deleted xml for unavailable image file {:s}'.format(imgpath))
                continue
            imageCounter += 1

            # make filenames unique: append directory index to filename. First xml file, then image
            new_imagename = imagename+'_'+str(subdir_counter)
            os.rename(os.path.join(r,filename), os.path.join(r,new_imagename+'.xml'))
            filename = new_imagename + '.xml'
            filewithpath = os.path.join(r, filename)
            os.rename(os.path.join(r,imagename + ext), os.path.join(r,new_imagename+ext))
            imagename = new_imagename

            try:
                parent = filewithpath.split('/')[-2]
            except IndexError:
                parent = filewithpath.split('\\')[-2]
            parent = parent.replace(' ', '')
            parser = xml.etree.ElementTree.XMLParser(encoding="utf-8")
            tree = xml.etree.ElementTree.parse(filewithpath, parser = parser)
            root = tree.getroot()

            imglist += parent + imagename + postfix + '\n'
            roiCounter = 0
            # print(r+'/'+imagename + ext)
            # im = cv2.imread(os.path.join(r, imagename + ext))
            # assert im is not None
            imagebrands = []
            intersection = False
            for obj in root.findall('object'):
                brand = obj.find('name').text.encode('utf-8').lower()
                brand = brand.decode('utf-8')
                #brand = str(brand)
                if brand == "1.fcköln":
                    brand = "fckoeln"
                if brand == "adidas3":
                    brand = "adidas-text"
                if "adidas4" in brand:
                    brand = brand.replace("adidas4", "adidas3")
                if brand == "aluratek":
                    brand = "aluratek-symbol"
                if brand == "apecase":
                    brand = "apecase-symbol"
                if brand == "apecase-teilsichtbar":
                    brand = "apecase-symbol-teilsichtbar"
                if brand == "armitron1":
                    brand = "armitron"
                if brand == "audi":
                    brand = "audi-symbol"
                if brand == "b":
                    brand = "bridgestone"
                if brand == "basf-symbol":
                    brand = "basf"
                if "bertha1" in brand:
                    brand = brand.replace("bertha1", "bertha")
                if "boing" in brand:
                    brand = brand.replace("boing", "boeing")
                if "budweiser1" in brand:
                    brand = brand.replace("budweiser1", "budweiser")
                if "budweiser2" in brand:
                    brand = brand.replace("budweiser2", "budweiser")
                if brand == "budweiser-b":
                    brand = "budweiser-b-symbol"
                if brand == "budweiser-teilsichtbar":
                    brand = "budweiser-symbol-teilsichtbar"
                if "bundweiser" in brand:
                    brand = brand.replace("bundweiser", "budweiser")
                if brand == "burgerking":
                    brand = "burgerking-symbol"
                if brand == "burgerking-teilsichtbar":
                    brand = "burgerking-symbol-teilsichtbar"
                if "canon1" in brand:
                    brand = brand.replace("canon1", "canon")
                if "canon2" in brand:
                    brand = brand.replace("canon2", "canon")
                if "cartier1" in brand:
                    brand = brand.replace("cartier1", "cartier")
                if "caterpillar1" in brand:
                    brand = brand.replace("caterpillar1", "caterpillar")
                if brand == "chevrolet1":
                    brand = brand.replace("chevrolet1", "chevrolet")
                if brand == "citroen":
                    brand == "citroen-symbol"
                if brand == "colgate1":
                    brand = "colgate"
                if "dadone" in brand:
                    brand = brand.replace("dadone", "danone")
                if brand == "cvs-symbol" or brand == "cvs-logo":
                    brand = "cvspharmacy"
                if brand == "danone1":
                    brand = "danone"
                if "fils" in brand:
                    brand = brand.replace("fils", "fila")
                if brand == "google":
                    brand = "google-symbol"
                if brand == "gucci1":
                    brand = "gucci"
                if brand == "gucci logo":
                    brand = "gucci-symbol"
                if "heineke" in brand:
                    brand = brand.replace("heineke", "heineken")
                if brand == "hersheys1":
                    brand = "hersheys"
                if brand == "hungry jacks logo":
                    brand = "hungry jacks-symbol"
                if "hyundri" in brand:
                    brand = brand.replace("hyundri", "hyundai")
                if "kellogg`s-k" in brand:
                    brand = brand.replace("kellogg`s-k", "kellogg`s-symbol")
                if "kia-logo" in brand:
                    brand = brand.replace("kia-logo", "kia")
                if brand == "lego":
                    brand = "lego-symbol"
                if brand == "lego-teilsichtbar":
                    brand = "lego-symbol-teilsichtbar"
                if "louis vuitton2" in brand:
                    brand = brand.replace("louis vuitton2", "louisvuitton")
                if brand == "mastercard1":
                    brand = "mastercard"
                if brand == "mcdonalds":
                    brand = "mcdonalds-symbol"
                if brand == "mcdonalds-teilsichtbar":
                    brand = "mcdonalds-symbol-teilsichtbar"
                if brand == "mercedes" or brand == "mercedes-logo":
                    brand = "mercedesbenz-symbol"
                if brand == "mercedes-schrift" or brand == "mercedes-schriftzug":
                    brand = "mercedesbenz-schriftzug" # !!!!!
                if brand == "mercedes-teilsichtbar":
                    brand = "mercedesbenz-symbol-teilsichtbar"
                if brand == "nestle1" or brand == "nestle2":
                    brand = "nestle"
                if brand == "nike":
                    brand = "nike-symbol"
                if "nikelogo" in brand:
                    brand = brand.replace("nikelogo", "nike")
                if brand == "lego1":
                    brand = "lego"
                if brand == "nivea1":
                    brand = "nivea"
                if brand == "olympia":
                    brand = "olympicgames"
                if brand == "pizzahut-logo":
                    brand = "pizzahut"
                if "ruffels" in brand:
                    brand = brand.replace("ruffels", "ruffles")
                if brand == "the home depot1" or brand == "the home depot-logo":
                    brand = "thehomedepot"
                if "vl" in brand:
                    brand = brand.replace("vl", "louisvuitton")
                if brand == "volksbank":
                    brand = "volksbank-symbol"
                if brand == "ströker":
                    brand = "stroeer"
                if brand == "görtz":
                    brand = "goertz"
                if "schriftzug" in brand:
                    brand = brand.replace("-schriftzug", "-text") # !!!!
                if "schrift" in brand:
                    brand = brand.replace("-schrift", "-text") # !!!!
                if "teilsichtbar" in brand:
                    brand = brand.replace("-teilsichtbar", "")
                    obj.find('truncated').text = str(1)
                if "logo" in brand:
                    brand = brand.replace('logo', 'symbol')
                if "`" in brand:
                    brand = brand.replace("`", "")
                if "." in brand:
                    brand = brand.replace(".", "")
                brand = brand.replace(" ", "")


                if brand == "chanel":
                    brand = "chanel-text"
                if brand == "chanel-symbol":
                    brand = "chanel"

                if brand == "citroen":
                    brand = "citroen-text"
                if brand == "citroen-symbol":
                    brand = "citroen"

                if brand == "mcdonalds":
                    brand = "mcdonalds-text"
                if brand == "mcdonalds-symbol":
                    brand = "mcdonalds"

                if brand == "mercedesbenz":
                    brand = "mercedes-text"
                if brand == "mercedesbenz-symbol":
                    brand = "mercedes"

                if brand == "nike-symbol":
                    brand = "nike"

                if brand == "porsche":
                    brand = "porsche-text"
                if brand == "porsche-symbol":
                    brand = "porsche"

                if brand == "unicef-symbol":
                    brand = "unicef"

                if brand == "vodafone-symbol":
                    brand = "vodafone"

                roiname = parent + "_" +imagename + '_' + str(roiCounter)

                if roiname == "H&M_img000198_0" or roiname == "H&M_img000252_4":
                    brand = "adidas3"
                if (
                        roiname == "nivea_img000135_0" or roiname == "nivea_img000180_5" or
                        roiname == "red bull_img000292_2" or roiname == "red bull_img000292_9" or
                        roiname == "red bull_img000323_3" or roiname == "red bull_img000563_2" or
                        roiname == "red bull_img000563_4"
                   ):
                    brand = "adidas-text"
                if brand == "adidas" or brand == "adidas-symbol" or roiname == "adidas_img000419_0":
                    brand = "adidas1"
                if roiname == "adidas_img000023_0":
                    brand = "adidas3"
                if brand == "amazon-text":
                    brand = "amazon"
                if roiname == "boeing_img000039_2" or roiname == "boeing_img000043_1":
                    brand = "amazon"
                if brand == "americanexpress1":
                    brand = "americanexpress"
                if roiname == "BMW_img000103_2":
                    brand = "audi-symbol"
                if brand == "basf-symbol":
                    brand = "basf"
                if roiname == "volkswagen_img000419_2":
                    brand = "beatsaudio"
                if roiname == "bionade_img000097_2":
                    brand = "bionade-symbol"
                if roiname == "bosch_img000070_0" or brand == "bosch":
                    brand = "bosch-text"
                if roiname == "airhawk_img000030_0":
                    brand = "airhawk"
                if brand == "bud" or brand == "budweiser":
                    brand = "budweiser-text"
                if (roiname == "budweiser_img000008_5" or roiname == "budweiser_img000008_6" or roiname == "budweiser_img000008_7" or
                    roiname == "budweiser_img000008_8" or roiname == "budweiser_img000009_0" or roiname == "budweiser_img000177_5" or
                    roiname == "budweiser_img000177_6" or roiname == "budweiser_img000177_7" or roiname == "budweiser_img000202_0" or
                    roiname == "budweiser_img000210_3" or roiname == "budweiser_img000210_4" or roiname == "budweiser_img000376_4" or
                    roiname == "budweiser_img000376_5" or roiname == "budweiser_img000376_7"):
                    brand = "budweiser-text"
                if roiname == "burger king_img000172_0" or roiname == "McDonalds_img000594_1":
                    brand = "burgerking-text"
                if roiname == "burger king_img000131_2" or roiname == "burger king_img000420_1" or roiname == "burger king_img000166_3":
                    brand = "burgerking-symbol"
                if brand == "burkler":
                    brand = "buckler"
                if roiname == "FedEx_img000230_0":
                    brand = "fedex"
                if roiname == "heineken_img000045_51":
                    brand = "heineken"
                if roiname == "intel_img000073_0":
                    brand = "intel"
                if roiname == "netflix_img000115_0":
                    brand = "netflix"
                if roiname == "nivea_img000128_3":
                    brand = "nivea"
                if roiname == "Pampers_img000016_1" or roiname == "Pampers_img000144_2":
                    brand = "pampers"
                if roiname == "philips_img000069_1" or roiname == "philips_img000155_8":
                    brand = "philips"
                if roiname == "rolex_img000008_0":
                    brand = "rolex"
                if roiname == "sony_img000014_0":
                    brand = "sony"
                if roiname == "volkswagen_img000042_5" or roiname == "volkswagen_img000105_1":
                    brand = "vw"
                if roiname == "chanel_img000000_11" or roiname == "chanel_img000000_12":
                    brand = "chanel"
                if roiname == "hyundai_img000146_2" or roiname == "hyundai_img000199_0" or roiname == "panasonic_img000143_2" or roiname == "target_img000091_1":
                    brand = "chevrolet-symbol"
                if roiname == "hyundai_img000526_1":
                    brand = "citroen"
                if roiname == "esso_img000182_0" or roiname == "red bull_img000194_1" or roiname == "esso_img000101_3" or roiname == "esso_img000107_1":
                    brand = "esso-text"
                if roiname == "red bull_img000298_2":
                    brand = "esso-symbol"
                if roiname == "hyundai_img000146_0" or roiname == "kia_img000190_4" or roiname == "starbucks_img000181_3":
                    brand = "honda-symbol"
                if roiname == "kia_img000190_5":
                    brand = "honda"
                if brand == "coca-cola1":
                    brand = "coca-cola"
                if brand == "coke1":
                    brand = "coke"
                if brand == "copyofamcrest-symbol":
                    root.remove(obj)
                    continue
                if brand == "corona":
                    brand = "corona-text"
                if brand == "costco":
                    brand = "costco-text"
                if brand == "cvs":
                    brand = "cvspharmacy"
                if brand == "esso-symbol":
                    brand = "esso"
                if brand == "firelli":
                    brand = "pirelli"
                if brand == "ford":
                    brand = "ford-symbol"
                if brand == "frankfurt":
                    brand = "fcfrankfurt"
                if brand == "galeria":
                    brand = "galeriakaufhof"
                if brand == "google-symbol":
                    brand = "google-text"
                if roiname == "huawei_img000078_1":
                    brand = "google-symbol"
                if brand == "headshoulders":
                    brand = "headandhoulders"
                if brand == "heinekenn" or brand == "heinekenn-text" or brand == "heinekenn-würfel" or brand == "heinekenn1":
                    brand = "heineken"
                if brand == "honda":
                    brand = "honda-text"
                if brand == "hsbc":
                    brand = "hsbc-text"
                if brand == "huawei":
                    brand = "huawei-text"
                if roiname == "coca-cola_img000671_5" or roiname == "coca-cola_img000679_2" or roiname == "coca-cola_img000698_1" or roiname == "kia_img000236_4" or roiname == "shell_img000153_5":
                    brand = "hyundai-symbol"
                if brand == "infiniti":
                    brand = "infiniti-text"
                if brand == "intel-text":
                    brand = "intel"
                if brand == "kaiserslautern":
                    brand = "fckaiserslautern"
                if roiname == "kelloggs_img000090_3":
                    brand = "kelloggs-symbol"
                if roiname == "boeing_img000015_0":
                    brand = "boeing-text"
                if roiname == "audi_img000444_3" or roiname == "audi_img000444_4":
                    brand = "lexus-symbol"
                if brand == "lexus":
                    brand = "lexus-text"
                if brand == "madonalds":
                    brand = "mcdonalds"
                if brand == "malboro":
                    brand = "marlboro"
                if (roiname == "McDonalds_img000233_2" or roiname == "McDonalds_img000235_2" or roiname == "McDonalds_img000202_2" or
                    roiname == "pepsi_img000338_0" or roiname == "pizza hut_img000006_2" or roiname == "pizza hut_img000026_0" or
                    roiname == "pizza hut_img000073_2" or roiname == "pizza hut_img000118_2" or roiname == "shell_img000330_1" or
                    roiname == "shell_img000354_1" or roiname == "shell_img000361_1"):
                    brand = "mcdonalds-text"
                if brand == "mönchengladbach":
                    brand = "fcmoenchengladbach"
                if roiname == "red bull_img000344_10":
                    brand = "audi-symbol"
                if brand == "nissan" or roiname == "BMW_img000207_0" or roiname == "nissan_img000093_0":
                    brand = "nissan-symbol"
                if roiname == "kia_img000132_3" or roiname == "kia_img000171_3":
                    brand = "nissan-text"
                if brand == "olympia":
                    brand = "olympicgames"
                if brand == "opel":
                    brand = "opel-symbol"
                if roiname == "kia_img000100_11" or roiname == "kia_img000139_3":
                    brand = "opel-text"
                if brand == "oral-b":
                    brand = "oralb"
                if roiname == "Pampers_img000214_1":
                    brand = "sesamestreet"
                if brand == "panasonic":
                    brand = "panasonic-text"
                if brand == "pepsi" or brand == "pepsi-text1" or brand == "pepsi3":
                    brand = "pepsi-text"
                if roiname == "pepsi_img000360_1":
                    brand = "pepsi-symbol"
                if brand == "pizzahut-hut":
                    brand = "pizzahut-symbol"
                if roiname == "santander_img000090_3" or roiname == "shell_img000235_2" or roiname == "shell_img000451_0":
                    brand = "ferrari"
                if roiname == "esso_img000280_2" or roiname == "nissan_img000294_3" or roiname == "nissan_img000298_5" or roiname == "nissan_img000307_5" or roiname == "nissan_img000333_6":
                    brand = "renault-symbol"
                if brand == "rolex-krone":
                    brand = "rolex-symbol"
                if brand == "samsung1":
                    brand = "samsung"
                if roiname == "huawei_img000274_3":
                    brand = "huawei-text"
                if roiname == "kraft_img000113_0":
                    brand = "scania-symbol"
                if roiname == "volkswagen_img000625_4":
                    brand = "seat"
                if roiname == "budweiser_img000011_0" or roiname == "budweiser_img000011_1" or roiname == "budweiser_img000135_2":
                    brand = "budweiser-select-symbol"
                if brand == "shell" or roiname == "shell_img000405_1" or roiname == "shell_img000405_2":
                    brand = "shell-symbol"
                if brand == "shell-text1":
                    brand = "shell-text"
                if brand == "siemens":
                    brand = "siemens-text"
                if roiname == "esso_img000108_0" or roiname == "esso_img000158_1":
                    brand = "esso-text"
                if roiname == "costco_img000090_1" or brand == "starbuckscoffe" or brand == "starbuckscoffee":
                    brand = "starbucks-text"
                if roiname == "shell_img000354_4" or roiname == "shell_img000361_2" or brand == "starbucks-symbol+" or brand == "starbucks-symbol+teilsichtbar" or roiname == "starbucks_img000064_1":
                    brand = "starbucks-symbol"
                if roiname == "red bull_img000610_6" or roiname == "red bull_img000610_7":
                    brand = "subaru-symbol"
                if roiname == "nissan_img000333_8":
                    brand = "citroen"
                if brand == "suzuki":
                    brand = "suzuki-text"
                if roiname == "nissan_img000333_9":
                    brand = "citroen-text"
                if brand == "target1":
                    brand = "target"
                if brand == "t-mobile":
                    brand = "tmobile"
                if brand == "toronto":
                    brand = "fctoronto"
                if roiname == "toyota_img000015_2" or brand == "toyota-text1" or brand == "toyota1":
                    brand = "toyota-text"
                if brand == "tsv-münchen":
                    brand = "tsvmuenchen"
                if brand == "ups1":
                    brand = "ups"
                if brand == "visa-electron" or brand == "visa1":
                    brand = "visa"
                if roiname == "audi_img000216_5":
                    brand = "sparkasse"
                if roiname == "BMW_img000479_1":
                    brand = "toyota"
                if roiname == "kia_img000164_5":
                    brand = "volvo-symbol"
                if roiname == "gillette_img000371_1" or roiname == "gillette_img000371_2" or roiname == "gillette_img000371_3":
                    brand = "walmart-symbol"
                if brand == "walmart-neu":
                    brand = "walmart"
                if brand == "würth":
                    brand = "wuerth"
                if brand == "bochum":
                    brand = "fcbochum"
                if brand == "dresden":
                    brand = "fcdresden"
                if brand == "msvduisnurg":
                    brand = "msvduisburg"
                if roiname == "pizza hut_img000144_3":
                    brand = "pizzahut"
                if roiname == "audi_img000150_3" or roiname == "BMW_img000485_1":
                    brand = "deutschepost"
                if roiname == "Walmart_img000026_0":
                    brand = "walmart-symbol"
                if brand == "schöller":
                    brand = "schoeller"

                obj.find('name').text = brand
                imagebrands.append(brand)
                if args.wofl32 and brand in fl32_intersection:
                    intersection = True

                if args.roi:
                    bndbox = obj.find('bndbox')
                    x1 = int(bndbox[0].text)
                    y1 = int(bndbox[1].text)
                    x2 = int(bndbox[2].text)
                    y2 = int(bndbox[3].text)
                    roi = im[y1:y2, x1:x2]
                    brandspath = os.path.join(args.outpath, 'brandROIs')
                    folder = os.path.join(brandspath, brand)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    cv2.imwrite(os.path.join(folder, roiname + '.jpg'), roi)
                roiCounter += 1
                totalRoiCounter += 1

            tree.write(filewithpath, encoding="UTF-8")

            if intersection:
                continue
            roiCounter = 0
            for obj in root.findall('object'):

                labelBrand = obj.find('name').text.encode('utf-8').lower()
                if labelBrand == 'copy of amcrest-logo':
                    continue

                brand = imagebrands[roiCounter]
                roiCounter += 1

                bndbox = obj.find('bndbox')
                x1 = int(bndbox[0].text)
                y1 = int(bndbox[1].text)
                x2 = int(bndbox[2].text)
                y2 = int(bndbox[3].text)

                brandlist.append(brand)

    with open(os.path.join(args.outpath, 'brands.txt'), 'w') as f:
        for brand in set(brandlist):
            f.write(brand + '\n')

    print('Processed folders: {:d}'.format(subdir_counter))
    print('Processed rois: {:d}'.format(totalRoiCounter))
    print('Processed images: {:d}'.format(imageCounter))
    print('Processed brands: {:d}'.format(len(set(brandlist))))
    print('Unavailable image files: {:d}'.format(unavailableCounter))
