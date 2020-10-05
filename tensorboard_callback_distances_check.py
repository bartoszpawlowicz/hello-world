from identification.utils.utils import compute_eer, return_distances_as_array
import time
import numpy as np
import os
import io
import tensorflow as tf
import cProfile
import pstats
from PIL import Image
import pickle
import keras
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def profile(fnc):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        sortby = 'cumulative'
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.print_stats()
        print(ps.print_stats())
        return retval

    return inner


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width, channel = tensor.shape
    tensor_fixed = (tensor + 1) / 2
    image = Image.fromarray((tensor_fixed * 255).astype('uint8'))
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string, )


def prepare_image_summary(example_1, example_2, is_false_negative, name, epoch, writer, tag=''):
    example_1 = example_1[:, :, ::-1]
    example_2 = example_2[:, :, ::-1]
    example_sum = np.concatenate((example_1, example_2), axis=1)
    example_image = make_image(example_sum)
    if is_false_negative:
        mistake_type = "_False_Negatives/"
    else:
        mistake_type = "_False_Positives/"
    full_tag = ''.join((name, mistake_type, tag, '/'))
    image_summary = tf.Summary(value=[
        tf.Summary.Value(tag=full_tag,
                         image=example_image)])
    writer.add_summary(image_summary, epoch)
    writer.flush()


def log_histogram(writer, tag, values, step, bins=1000):
    # Convert to a numpy array
    values = np.array(values)

    min_val = float(0)
    max_val = float(1.5)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, density=True, bins=bins, range=(min_val, max_val))

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = min_val
    hist.max = max_val
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()


def log_plot(writer, tag, plot, epoch):

    buf = io.BytesIO()
    plot.savefig(buf, format='png')

    buf.seek(0)
    image = Image.open(buf)
    width, height = image.size
    image_string = buf.getvalue()
    buf.close()

    image_summary = tf.Summary.Image(height=height,
                                     width=width,
                                     colorspace=4,
                                     encoded_image_string=image_string)
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=image_summary)])
    writer.add_summary(summary, epoch)
    writer.flush()


def all_distances_np(validation_set, model, max_examples, name, epoch, writer):

    [x, y] = validation_set
    embeddings = model.predict(x)

    p_distances, n_distances, a_p_ids, a_n_ids = return_distances_as_array(y, embeddings, squared=False)

    eer, thresh, auc_score = compute_eer(p_distances, n_distances)

    if max_examples > 0:
        draw_all_distances_np(validation_set, p_distances, n_distances, a_p_ids, a_n_ids, thresh, max_examples, name, epoch, writer)

    return eer, thresh, auc_score, p_distances, n_distances


def draw_all_distances_np(validation_set, p_distances, n_distances, a_p_ids, a_n_ids, distance_threshold, max_examples, name, epoch, writer):

    [x, y] = validation_set
    fn_examples_list = []
    fp_examples_list = []

    for d, distance in enumerate(p_distances):
        if distance > distance_threshold:
            i, j = a_p_ids[d]
            score = round(float(distance), 1)
            fn_examples_list.append((score, i, j))

    for d, distance in enumerate(n_distances):
        if distance < distance_threshold:
            i, j = a_n_ids[d]
            score = round(float(distance), 1)
            fp_examples_list.append((score, i, j))

    sorted_fn_list = sorted(fn_examples_list, key=lambda tup: tup[0], reverse=True)
    for (score, i, j) in sorted_fn_list[0:max_examples]:
        prepare_image_summary(example_1=x[i], example_2=x[j], is_false_negative=True,
                              name=name, epoch=epoch, writer=writer, tag=str(epoch))

    sorted_fp_list = sorted(fp_examples_list, key=lambda tup: tup[0], reverse=False)
    for (score, i, j) in sorted_fp_list[0:max_examples]:
        prepare_image_summary(example_1=x[i], example_2=x[j], is_false_negative=False,
                              name=name, epoch=epoch, writer=writer, tag=str(epoch))


class CheckAllDistancesCheckpoint(keras.callbacks.Callback):
    def __init__(self, validation_set, tensorboard, period=5, draw_plot=False, distance_threshold=None,
                 auto_distance_threshold=True, max_examples=10):
        super().__init__()
        self.validation_set = validation_set
        self.period = period
        self.draw_plot = draw_plot
        self.distance_threshold = distance_threshold
        self.auto_distance_threshold = auto_distance_threshold
        self.max_examples = max_examples
        self.tensorboard = tensorboard

    # @profile
    def on_epoch_end(self, epoch, logs=None, force=False):

        super(CheckAllDistancesCheckpoint, self).on_epoch_end(epoch, logs)
        start = time.time()

        if self.period > 0:
            if epoch % self.period != 0:
                return

            name = "full_validation"

            eer, thresh, auc_score, positive_distances, negative_distances =\
                all_distances_np(
                    self.validation_set, self.model,
                    self.max_examples, name, epoch, self.tensorboard.writer)

            self.distance_threshold = thresh
            print("--------------------------")
            print("Batch Generator name:", name)
            print("EER:", str(eer))
            print("Threshold:", str(thresh))
            print("--------------------------")
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="accuracy_" + str(name) + "/accuracy_at_eer_thr", simple_value=1.-float(eer)),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="accuracy_" + str(name) + "/eer", simple_value=eer),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="accuracy_" + str(name) + "/auc", simple_value=auc_score),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            self.tensorboard.writer.flush()

            print("--------------------------")
            print("Batch Generator name:", name)
            print("NO Positive Distances:", len(positive_distances))
            print("NO Negative Distances:", len(negative_distances))

            pos_mean = np.mean(positive_distances)
            neg_mean = np.mean(negative_distances)
            diff_mean = neg_mean - pos_mean

            print("")
            print("Mean positive distance:", pos_mean)
            print("Mean negative distance:", neg_mean)
            print("Mean difference distance:", diff_mean)

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="distances_" + str(name) + "/positive_mean_distance",
                                 simple_value=pos_mean),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="distances_" + str(name) + "/negative_mean_distance",
                                 simple_value=neg_mean),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="distances_" + str(name) + "/mean_distance_difference",
                                 simple_value=diff_mean),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            self.tensorboard.writer.flush()

            pos_median = np.median(positive_distances)
            neg_median = np.median(negative_distances)
            diff_median = neg_median - pos_median

            print("")
            print("Median positive distance:", pos_median)
            print("Median negative distance:", neg_median)
            print("Median difference distance:", diff_median)

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="distances_" + str(name) + "/positive_median_distance",
                                 simple_value=pos_median),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="distances_" + str(name) + "/negative_median_distance",
                                 simple_value=neg_median),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="distances_" + str(name) + "/median_distance_difference",
                                 simple_value=diff_median),
            ])
            self.tensorboard.writer.add_summary(summary, epoch)
            self.tensorboard.writer.flush()

            print("--------------------------")

            if name == 'valid' and self.auto_distance_threshold:
                self.distance_threshold = np.mean([pos_median, neg_median])

            if self.draw_plot:

                try:

                    fig = plt.figure()
                    # val_range = (0, np.sqrt(2))
                    val_range = (0, 2)
                    plt.hist(positive_distances, alpha=0.5, color='b', density=True, range=val_range, bins=20)
                    plt.hist(negative_distances, alpha=0.5, color='r', density=True, range=val_range, bins=20)

                    log_plot(self.tensorboard.writer, "histograms/" + str(name), fig, epoch)

                    # log_histogram(self.tensorboard.writer, "positive_hist/" + str(name), positive_distances, epoch, 20)
                    # log_histogram(self.tensorboard.writer, "negative_hist/" + str(name), negative_distances, epoch, 20)
                    # log_histogram(self.tensorboard.writer, "combined_hist/" + str(name), positive_distances, 2 * epoch, 20)
                    # log_histogram(self.tensorboard.writer, "combined_hist/" + str(name), negative_distances, 2 * epoch + 1, 20)

                    plt.close(fig)

                except Exception as e:
                    print("Drawing plot error:", e)

        end = time.time()
        print("checkpoint time: " + str(end - start))
