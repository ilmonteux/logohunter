import os
import time
import argparse
import readline
readline.parse_and_bind("tab: complete")

from multiprocessing.pool import ThreadPool

from urllib.request import Request, urlopen, HTTPError
import ssl

def fetch_url(url_file):
    """
    Fetch url and download content to fileself.

    Args:
      url_file: (url, file) tuple. If file already exists, mark as downloaded and skip.
    Returns:
      bool: True if download successfull, False otherwise.
    """
    url, file_out = url_file
    if os.path.exists(file_out):
        return True

    req = Request(url)
    # add user-agent header to pass as real browser
    req.add_header('User-Agent', 'Mozilla/5.0 (Linux; Android 5.1.1; SM-G928X Build/LMY47X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.83 Mobile Safari/537.36')
    try:
        # use ssl._create_unverified_context to skip certificate verification
        page = urlopen(req, context=ssl._create_unverified_context()).read()
        with open(file_out, 'wb') as f:
            f.write(page)
        return True
    except:
        return False

def main(dir_litw):

    classes_all = []

    # in each folder, find urls.txt file and download URL in each line
    for folder in sorted(os.listdir(dir_litw), key=str.casefold):
        if not os.path.isdir(os.path.join(dir_litw, folder)):
            continue
        classes_all.append(folder)

        # no annotations in folder 0samples/
        if folder == '0samples':
            continue

        print(time.strftime("%H:%M:%S %Z"),'Downloading images in folder {}...'.format(folder), end='')

        with open(os.path.join(dir_litw, folder,'urls.txt'),'r', errors='ignore') as txtfile:
            start = time.time()

            img_ids, urls = zip(*[line.split('\t') for line in txtfile.readlines()])
            filepaths = [os.path.join(dir_litw, folder, 'img{}.jpg'.format(img_id)) for img_id in img_ids]

            results = ThreadPool(20).imap_unordered(fetch_url, zip(urls, filepaths))
            results = list(results)

            end = time.time()
            print('{} images in {:.1f} sec!'.format(sum(results), end-start))
    return classes_all

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
        '--dir_litw', type=str, default=os.path.join(os.path.pardir, 'data_litw', 'LogosInTheWild-v2', 'data'),
        help='path to Logos In The Wild data/ parent folder. Each subfolder contains a url.txt with links to images'
    )
    args = parser.parse_args()

    dir_litw = args.dir_litw

    main(dir_litw)
