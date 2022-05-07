import cairosvg
import matplotlib.pyplot as plt
import numpy as np
from .svg_utils import *
import sys
import PIL
from .svg_utils import *
import linecache
from io import BytesIO, StringIO
import torch
from torchvision import transforms
from threading import Thread, Lock
import numpy as np
import datetime

max_f3 = 400

def parallel_map(func, arglist, workers=16):
    """
    Map a function to a list of arguments in parallel.

    @param func: The function to map.
    @param arglist: The list of arguments to map.
    @param workers: The number of parallel workers to use.
    """
    results = [None] * len(arglist)

    def thread_prune(threads):
        """
        Prune all finished threads.
        """
        living = []
        for i in range(len(threads)):
            if threads[i].is_alive():
                living.append(threads[i])
            else:
                threads[i].join()

        return living

    threads = []
    lock = Lock()
    while len(arglist) > 0:
        if len(threads) < workers:
            idx = len(arglist) - 1
            threads.append(Thread(target=parallel_map_worker, args=(func, arglist.pop(-1), results, idx, lock)))
            threads[-1].start()
        else:
            threads = thread_prune(threads)
            sleep(0.01) # Yes, I know, busywait is bad

    for thread in threads:
        thread.join()

    return results

def parallel_map_worker(func, args, results, idx, lock):
    out = func(*args)
    lock.acquire()
    results[idx] = out
    lock.release()
    return

def render_clusters(svg_file, cluster_func=kmeans_centroids):
    """
    Render each cluster from an SVG file.

    @param svg_file: The SVG file to render. If using k_box_clustering, pass in the ABSOLUTE path to the SVG file since it needs the original raster image for the image dimensions.
    @param clusters: The clusters of SVG components. List of list of indicies
    @param cluster_func: The function to use for clustering.
    """

    segments, color, transforms = load_segments(svg_file)
    clusters, segment_centroids = cluster_func(segments, transforms, color, 12)

    render_svg(svg_file, segment_centroids)

    file = open(svg_file, 'r')
    lines = file.readlines()

    # Render each cluster
    for cluster in clusters:
        cluster_file = open('clusters_tmp_k_box.svg', 'w')
        # save lines for each segment to file and render
        cluster_file.write(lines[0])
        cluster_file.write(lines[1])
        for segment_idx in cluster:
            cluster_file.write(lines[segment_idx + 2])
        cluster_file.write(lines[-1])
        
        # Render the cluster
        cluster_file.close()
        render_svg('clusters_tmp_k_box.svg', segment_centroids[cluster])

def batch_render_clusters_correspondence(svg_files, svg_infos, sim, num_segments, cluster_func=kmeans_centroids):
    # creates dir if doesn't exist
    if not os.path.exists('/dev/shm/CAIN_TMP'):
        os.mkdir('/dev/shm/CAIN_TMP')
    workers = []
    all_renders = [None] * len(svg_files)
    # print(len(svg_files))
    # print(sim.shape[0])
    lock = Lock()
    # print(sim.shape)
    for i in range(0, len(svg_files)):
        worker = Thread(target=worker_render_cluster_corr, args=(svg_files[i][0], svg_files[i][len(svg_files[i])-1], svg_infos[i][0], svg_infos[i][len(svg_files[i])-1], sim[i][:num_segments[i][0], :num_segments[i][len(svg_files[i])-1]], cluster_func, all_renders, i, lock))
        worker.start()
        workers.append(worker)
        # render_clusters_correspondence(svg_files[i][0], svg_files[i][2], svg_infos[i][0], svg_infos[i][2], sim[i][:num_segments[i][0], :num_segments[i][2]], cluster_func, all_renders, i)

    for worker in workers:
        worker.join()

    masks = torch.stack(all_renders, dim=0)
    print('Batched render complete')
    return masks

def worker_render_cluster_corr(svg_frame1, svg_frame3, svg_frame1_info, svg_frame3_info, corr, cluster_func, all_renders, idx, lock):
    segments1, color1, transforms1 = svg_frame1_info
    segments3, color3, transforms3 = svg_frame3_info

    clusters1, segment_centroids1 = cluster_func(segments1, transforms1, color1, 3)
    
    lock.acquire()
    svg_file1 = open(svg_frame1, 'r')
    lines1 = svg_file1.readlines()
    svg_file1.close()
    svg_file3 = open(svg_frame3, 'r')
    lines3 = svg_file3.readlines()
    svg_file3.close()
    lock.release()

    cluster3_prerender, prerender_threads = parallel_prerender_start(lines3)
    cluster1_render_argslist = [(cluster, lines1) for cluster in clusters1]
    cluster1_renders = torch.stack(parallel_map(render_cluster_mask, cluster1_render_argslist), dim=0)

    # if (len(lines3) - 3 != corr[0].shape[0]):
        # print("BAD")
        # print(svg_frame1, svg_frame3)
        # print(len(lines3) - 3, corr[0].shape[0])
        # sys.exit()

    # cluster3_prerender = parallel_prerender(lines3)
    best_c1_per_c3 = torch.argmax(corr, dim=0)
    best_c1_per_c3_opacity = torch.max(corr, dim=0)[0]
    inverse_best = [[] for i in range(len(lines1) -3)]
    # print(corr.shape)
    # print(best_c1_per_c3.shape)
    # print(len(lines3) - 3)
    for i in range(min(len(lines3) - 3, max_f3)):
        inverse_best[best_c1_per_c3[i]].append(i)

    # inverse_best = np.array(inverse_best, dtype='object')
    summed_correspond = torch.stack([torch.sum(corr[clusters1[c]], dim=0) for c in range(len(clusters1))], dim=0)
    #normalize summed_correspond by max value
    summed_correspond = summed_correspond / torch.max(summed_correspond, dim=0)[0]

    # print(summed_correspond.shape)
    parallel_prerender_collect(prerender_threads)
    c3_render_argslist = []
    for cluster_idx in range(len(clusters1)):
        # print(clusters1[cluster_idx])
        # print(best_c1_per_c3)
        # print(best_c1_per_c3.shape)
        c3_idxes = []
        inverse_select = [inverse_best[i] for i in clusters1[cluster_idx]]
        for c3_idx in inverse_select:
            if isinstance(c3_idx, np.ndarray):
                c3_idx = c3_idx.tolist()
            c3_idxes += c3_idx

        c3_render_argslist.append((c3_idxes, lines3, best_c1_per_c3_opacity[c3_idxes], cluster3_prerender))

        # c3_render_argslist = [(np.arange(len(lines3) - 3)[clusters1[cluster_idx] == best_c1_per_c3], lines3, summed_correspond[cluster_idx][clusters1[cluster_idx] == best_c1_per_c3], cluster3_prerender) for cluster_idx in range(len(clusters1))]
    c3_render_list = parallel_map(render_cluster_mask, c3_render_argslist, workers=16)
    # c3_render_list = [render_cluster_mask(np.arange(len(lines3) - 3), lines3, summed_correspond[cluster_idx], cluster3_prerender) for cluster_idx in range(len(clusters1))]
    cluster3_renders = torch.stack(c3_render_list, dim=0)

    res = torch.stack([cluster1_renders, cluster3_renders], dim=0)
    lock.acquire()
    all_renders[idx] = res
    lock.release()

def render_clusters_correspondence(svg_frame1, svg_frame3, svg_frame1_info, svg_frame3_info, correspondences, cluster_func=kmeans_centroids, place_result=None, place_idx=None, mutex=None):
    """
    Render all clusters in SVG file (frame1) and all corresponding clusters in frame3
    @param svg_frame1: svg file for frame 1
    @param svg_frame3: svg file for frame 3
    @param correspondences: torch tensor (#segmments in frame1, #segments in frame3)
    """
    # creates dir if doesn't exist
    if not os.path.exists('/dev/shm/CAIN_TMP'):
        os.mkdir('/dev/shm/CAIN_TMP')

    segments1, color1, transforms1 = svg_frame1_info
    segments3, color3, transforms3 = svg_frame3_info

    clusters1, segment_centroids1 = cluster_func(segments1, transforms1, color1, 2)
    
    svg_file1 = open(svg_frame1, 'r')
    lines1 = svg_file1.readlines()
    cluster1_renders = torch.stack([render_cluster_mask(cluster, lines1) for cluster in clusters1], dim=0)

    svg_file3 = open(svg_frame3, 'r')
    lines3 = svg_file3.readlines()
    if (len(lines3) - 3 != correspondences[0].shape[0]):
        print("BAD")
        print(svg_frame1, svg_frame3)
        print(len(lines3) - 3, correspondences[0].shape[0])
        sys.exit()
    # print("Correspondences shape", correspondences.shape)
    # print("Clusters1", len(clusters1))
    # print("Filename1", svg_frame1)
    # print("Filename3", svg_frame3)
    # cluster3_prerender = all_segments_prerender(lines3)
    cluster3_prerender = parallel_prerender(lines3)
    # print("Clusters3", len(cluster3_prerender))
    # TODO: fix dim misalignment
    summed_correspond = torch.stack([torch.sum(correspondences[clusters1[c]], dim=0) for c in range(len(clusters1))], dim=0)
    #normalize summed_correspond by max value
    summed_correspond = summed_correspond / torch.max(summed_correspond, dim=0)[0]
    # print(summed_correspond.shape)
    c3_render_list = [render_cluster_mask(np.arange(len(lines3) - 3), lines3, summed_correspond[cluster_idx], cluster3_prerender) for cluster_idx in range(len(clusters1))]
    # print("Cluster3 renders", len(c3_render_list))
    # for c in c3_render_list:
        # print(type(c))
        # print(c.shape)
    cluster3_renders = torch.stack(c3_render_list, dim=0)

    res = torch.stack([cluster1_renders, cluster3_renders], dim=0)
    if place_result is not None:
        place_result[place_idx] = res
    else:
        return res


def color_to(line, color):
    #find color idx
    color_idx = line.index('#')
    # replace color
    line = line[:color_idx] + color + line[color_idx + 7:]
    return line

def prerender_worker(svg_file_lines, worker_queue, lock, all_images):
    """
    Worker function for parallel rendering.
    """
    frame_size = (1, 1, 240, 424)
    while True:
        lock.acquire()
        if len(worker_queue) == 0:
            lock.release()
            return
        idx = worker_queue.pop()
        # print("Worker working on", idx)
        lock.release()
        # get date string with milliseconds
        date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        f_name = '/dev/shm/CAIN_TMP/preworker_tmp' + str(idx) + date_string + '.svg'
        file = open(f_name, 'w')
        file.write(svg_file_lines[0])
        file.write(svg_file_lines[1])
        line = color_to(svg_file_lines[idx + 2], '#FFFFFF')
        file.write(line)
        file.write(svg_file_lines[-1])
        file.seek(0)
        image = render_svg(f_name)
        image = transforms.ToTensor()(image)[0].cuda()
        image = transforms.functional.crop(image, 0, 0, frame_size[2], frame_size[3])
        # delete fname
        os.remove(f_name)
        lock.acquire()
        all_images[idx] = image
        lock.release()

def parallel_prerender_start(svg_file_lines, workers=16):
    """
    Prerender all segments in an SVG file.
    @param svg_file_lines: The SVG file to render.
    @param workers: The number of threads to use.
    """
    all_images = [None] * (len(svg_file_lines) - 3)
    lock = Lock()
    worker_queue = list(np.arange(min(len(svg_file_lines) - 3, max_f3))) #no more than max_f3 segments will be rendered
    threads = []
    for i in range(min(workers, len(svg_file_lines) - 3)):
        threads.append(Thread(target=prerender_worker, args=(svg_file_lines, worker_queue, lock, all_images)))
        threads[-1].start()

    # for i in range(len(threads)):
        # threads[i].join()

    # print("Parallel prerender complete", len(all_images))
    return all_images, threads

def parallel_prerender_collect(threads):
    """
    Collect all prerendered images from threads.
    @param threads: The list of threads to join.
    """
    for i in range(len(threads)):
        threads[i].join()
    return threads



def all_segments_prerender(svg_file_lines):
    """
    Renders out each segment in SVG file as masks to differnt channels of an image tensor.
    @param svg_file_lines: lines of SVG file whose segments are individually rendered
    @return: list of masks (1, H, W)
    """
    frame_size = (1, 1, 240, 424)
    segments_rendered = []
    for l in range(len(svg_file_lines) - 3):
        cluster_file = open('clusters_tmp_k_box.svg', 'w')
        cluster_file.write(svg_file_lines[0])
        cluster_file.write(svg_file_lines[1])
        line = color_to(svg_file_lines[l + 2], '#FFFFFF')
        cluster_file.write(line)
        cluster_file.write(svg_file_lines[-1])
        cluster_file.close()
        img_tensor = transforms.ToTensor()(render_svg('clusters_tmp_k_box.svg'))[0].cuda()
        image = transforms.functional.crop(img_tensor, 0, 0, frame_size[2], frame_size[3]) # hard coded an arbitrary 240p size
        segments_rendered.append(image)
    return segments_rendered


def render_cluster_mask(cluster, svg_file_lines, opacities=None, cluster_prerender=None):
    """
    Rneder out a cluster of segments as a single channel mask.
    @param cluster: List of segment indicies
    @param svg_file_lines: SVG files read lines
    @param opactities: Tensor of opactities for each segment. Will set requires_grad=True
    @return torch tensor of rendered image
    """
    cluster_full = None#torch.tensor(np.zeros((img_np.shape[0], img_np.shape[1], 3)))
    # if opactities is not None:
    #     print(opactities.shape)
    # print(cluster)
    frame_size = (1, 1, 240, 424)

    if cluster_prerender is None:
        cluster_str = '_' if len(cluster) == 0 else str(cluster[0])
        date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        f_name = '/dev/shm/CAIN_TMP/r_cluster_mask' + cluster_str + date_string + '.svg'
        cluster_file = open(f_name, 'w')
        cluster_file.write(svg_file_lines[0])
        cluster_file.write(svg_file_lines[1])
        for i in cluster:
            line = color_to(svg_file_lines[i + 2], '#FFFFFF')
            cluster_file.write(line)
        cluster_file.write(svg_file_lines[-1])
        cluster_file.close()
        cluster_full = transforms.ToTensor()(render_svg(f_name))[0].cuda()
        cluster_full = transforms.functional.crop(cluster_full, 0, 0, frame_size[2], frame_size[3]) # hard coded an arbitrary 240p size
        os.remove(f_name)
    else:

    # for segment_idx in cluster:
        # if opacities is not None and (opacities[segment_idx]) < (torch.max(opacities) / 10):
            # continue
        # if cluster_prerender is None:
            # cluster_file = open('clusters_tmp_k_box.svg', 'w')
            # # cluster_file = BytesIO()
            # # save lines for each segment to file and render
            # cluster_file.write(svg_file_lines[0])
            # cluster_file.write(svg_file_lines[1])
            # # for segment_idx in cluster:
            # line = color_to(svg_file_lines[segment_idx + 2], '#FFFFFF')
            # cluster_file.write(line)
            # cluster_file.write(svg_file_lines[-1])
        
            # # Render the segment
            # cluster_file.close()
            # img = render_svg('clusters_tmp_k_box.svg')
            # img_tensor = transforms.ToTensor()(img)[0].cuda()
            # img_tensor = transforms.functional.crop(img_tensor, 0, 0, frame_size[2], frame_size[3]) # hard coded an arbitrary 240p size
            # img_tensor = img_tensor if opacities is None else img_tensor * opacities[segment_idx]
            # cluster_full = img_tensor if cluster_full is None else cluster_full + img_tensor
        # else:
        for i, segment_idx in enumerate(cluster):
            img_tensor = cluster_prerender[segment_idx]
            img_tensor = img_tesnor if opacities is None else img_tensor * opacities[i]
            cluster_full = img_tensor if cluster_full is None else cluster_full + img_tensor
    if cluster_full is None:
        return torch.zeros((frame_size[2], frame_size[3])).cuda()

    return cluster_full


def render_svg(file, points=None, show=False):
    png_image = BytesIO()
    cairosvg.svg2png(url=file, write_to=png_image)
    img = PIL.Image.open(png_image)
    img_np = np.array(img)

    if points is not None:
        # plot a point at each centroid
        for point in points:
            x, y = point
            # print(img_np.shape)
            img_np[min(int(y), img_np.shape[0] - 1), min(int(x), img_np.shape[1] - 1), :] = 0
            img_np[min(int(y), img_np.shape[0] - 1), min(int(x), img_np.shape[1] - 1), 0] = 255

    if show:
        plt.imshow(img_np)
        plt.show()

    return img


if __name__ == '__main__':
    render_clusters(sys.argv[1])
