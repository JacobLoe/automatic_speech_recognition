# source $HOME/tmp/deepspeech-venv/bin/activate

from deepspeech import Model
import wave
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
import os
import glob
import shutil

VERSION = '20200719'


def extract_wav_from_video(video_path, movie_id):
    # extracts the audio of a video-file to a .wav-file in /tmp
    audio_path = '/tmp/'+str(movie_id)+'_audio.wav'

    # -y -> overwrite output files
    # -i -> inputfile
    # outputfile options
    # -ab -> audio bitrate
    # -ac -> audio channels
    # -ar -> audio sampling rate in Hz
    # -vn -> disable video ??
    command = "ffmpeg -y -loglevel quiet -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(video_path, audio_path)
    subprocess.call(command, shell=True)


def metadata_to_string(metadata, min_confidence):
    conf = np.abs(metadata.confidence)
    if conf > min_confidence:
        return ''.join(token.text for token in metadata.tokens) + " ({0})".format(conf)
    else:
        return "SIL"


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0

    conf = np.abs(metadata.confidence)

    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)     # relative to chunks
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def process_wavefile(ds, wavefile, segment_length_ms, min_confidence):
    transcript = []
    with wave.open(wavefile, 'rb') as ain:
        framerate = ain.getframerate()
        nframes = ain.getnframes()
        audio_length_ms = nframes * (1. / framerate) * 1000

        for start_ms in tqdm(range(0, int(audio_length_ms), segment_length_ms)):
            ain.setpos(int(start_ms / 1000. * framerate))
            chunkData = np.frombuffer(ain.readframes(int(segment_length_ms / 1000. * framerate)), np.int16)

            words = metadata_to_string(ds.sttWithMetadata(chunkData, 1).transcripts[0], min_confidence)

            transcript.append((start_ms, start_ms + segment_length_ms, words))
    return transcript


def write_transcript_to_file(features_path, transcript):
    with open(os.path.join(features_path, 'audio.csv'), 'w') as f:
        for t in transcript:
            line = "{0} {1} {2}\n".format(t[0], t[1], t[2])
            f.write(line)


def transcribe(deepspeech_model, deepspeech_scorer, list_videos_path, movie_ids, features_path, segment_length_ms, min_confidence):
    ds = Model(deepspeech_model)
    ds.enableExternalScorer(deepspeech_scorer)

    # repeat until all movies are transcribed correctly
    done = 0
    while done < len(list_videos_path):
        for v_path, mid in tqdm(zip(list_videos_path, movie_ids), total=len(list_videos_path)):
            video_name = os.path.split(v_path)[1]
            # the directory for the results
            f_path = os.path.join(features_path, str(mid)+'.automatic_speech_recognition')
            done_file_path = os.path.join(f_path, '.done')

            if not os.path.isdir(f_path):   # check if the directory for the movie exists
                print('audio is being transcribed for {}'.format(video_name))
                os.makedirs(f_path)     # create new directory

                extract_wav_from_video(v_path, mid)
                audio_path = '/tmp/' + str(mid) + '_audio.wav'
                transcript = process_wavefile(ds, audio_path, segment_length_ms, min_confidence)
                write_transcript_to_file(f_path, transcript)

                # create a hidden file to signal that the optical flow for a movie is done
                # write the current version of the script in the file
                with open(done_file_path, 'a') as d:
                    d.write(VERSION)
                done += 1  # count the instances of the optical flow done correctly
            # do nothing if a .done-file exists and the versions in the file and the script match
            elif os.path.isfile(done_file_path) and open(done_file_path, 'r').read() == VERSION:
                done += 1  # count the instances of the optical flow done correctly
                print('optical flow was already done for {}'.format(video_name))
            # if the folder already exists but the .done-file doesn't, delete the folder
            elif os.path.isfile(done_file_path) and not open(done_file_path, 'r').read() == VERSION:
                shutil.rmtree(f_path)
                print('versions did not match for {}'.format(video_name))
            elif not os.path.isfile(done_file_path):
                shutil.rmtree(f_path)
                print('optical flow was not done correctly for {}'.format(video_name))


def main(deepspeech_model, deepspeech_scorer, videos_path, features_path, segment_length_ms, min_confidence):
    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    # create a list of ids for every movie
    movie_ids = list(range(len(list_videos_path)))

    transcribe(deepspeech_model, deepspeech_scorer, list_videos_path, movie_ids, features_path, segment_length_ms, min_confidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_path', help='path to the video to be transcribed')
    parser.add_argument('features_path', help='path to the directory where the results are saved')
    parser.add_argument('--deepspeech_model', default='../deepspeech-0.7.4-models.pbmm', help='path to a deepspeech model')
    parser.add_argument('--deepspeech_scorer', default='../deepspeech-0.7.4-models.scorer', help='path to a deepspeech scorer')
    parser.add_argument('--segment_length_ms', type=int, default=3000, help='')
    parser.add_argument('--min_confidence', type=float, default=10.0, help='')
    args = parser.parse_args()

    main(args.deepspeech_model, args.deepspeech_scorer, args.videos_path, args.features_path, args.segment_length_ms, args.min_confidence)
