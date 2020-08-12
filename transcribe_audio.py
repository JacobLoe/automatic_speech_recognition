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
import hashlib
from idmapper import TSVIdMapper

VERSION = '20200812'
EXTRACTOR = 'automatic_speech_recognition'


def extract_wav_from_video(video_path, movie_id):
    # extracts the audio of a video-file to a .wav-file in /tmp
    audio_path = '/tmp/'+str(movie_id)+'_audio.wav'

    # -y -> overwrite output files
    # -i -> inputfile
    # outputfile options
    # -ab -> audio bitrate
    # -ac -> audio channels
    # -ar -> audio sampling rate in Hz  # 44100
    # -vn -> disable video ??
    command = "ffmpeg -y -loglevel quiet -i {0} -ab 160k -ac 1 -ar 16000 -vn {1}".format(video_path, audio_path)
    subprocess.call(command, shell=True)


def words_from_candidate_transcript(metadata, start_ms, segment_lenght, min_confidence):
    word = ""
    word_list = []
    word_start_time = 0

    conf = np.abs(metadata.confidence)  # this results in every word having the same confidence
    if conf >= min_confidence:
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
                each_word["start_time"] = start_ms + int(round(word_start_time, 4)*1000)                    # time in milliseconds,
                each_word["duration"] = int(round(word_duration, 4)*1000)
                each_word["confidence"] = conf

                word_list.append(each_word)
                # Reset
                word = ""
                word_start_time = 0
    else:
        each_word = dict()
        each_word["word"] = 'SIL'
        each_word["start_time"] = start_ms
        each_word["duration"] = segment_lenght
        each_word["confidence"] = conf
        word_list.append(each_word)

    return word_list


def process_wavefile(ds, wavefile, segment_length_ms, min_confidence):
    transcript = []
    with wave.open(wavefile, 'rb') as ain:
        framerate = ain.getframerate()
        nframes = ain.getnframes()
        audio_length_ms = nframes * (1. / framerate) * 1000

        for start_ms in range(0, int(audio_length_ms), int(segment_length_ms)):

            ain.setpos(int(start_ms / 1000. * framerate))
            chunkData = np.frombuffer(ain.readframes(int(segment_length_ms / 1000. * framerate)), np.int16)

            words = words_from_candidate_transcript(ds.sttWithMetadata(chunkData, 1).transcripts[0], start_ms, segment_length_ms, min_confidence)

            transcript.append((start_ms, start_ms + segment_length_ms, words))
    return transcript


def get_overlapping_segments(ds, wavefile, transcript, overlap_length_ms, min_confidence):
    new_transcript = []
    with wave.open(wavefile, 'rb') as ain:
        framerate = ain.getframerate()
        nframes = ain.getnframes()
        audio_length_ms = nframes * (1. / framerate) * 1000
        for begin_segment, end_segment, segment in transcript:
            begin_overlap = end_segment-overlap_length_ms/2
            if begin_overlap+overlap_length_ms >= audio_length_ms:
                break
            ain.setpos(int(begin_overlap/1000. * framerate))
            chunkData = np.frombuffer(ain.readframes(int(overlap_length_ms / 1000. * framerate)), np.int16)
            segment_overlap = words_from_candidate_transcript(ds.sttWithMetadata(chunkData, 1).transcripts[0], begin_overlap, overlap_length_ms, min_confidence)

            new_transcript.append((begin_segment, end_segment, segment))
            new_transcript.append(segment_overlap)

    return new_transcript


def process_transcript(transcript, timestamp_threshold):
    new_transcript = []
    i = 0
    while i < len(transcript):
        # get the timestamps and words (and their timestamps) of the current segment
        begin_segment, end_segment, segment = transcript[i]
        if transcript[i+1]:     # check whether a overlap exists
            ov_segment = transcript[i+1]    # get the words and timestamps from the overlap
            # if ov_segment[0]['confidence'] < min_confidence: # check if the confidence of the overlap is higher than the required confidence
            #     break
            line = ' '  # start an empty line
            for s in segment:   # for every word in the segment
                line = line+' '+s['word']   # add the current word to the line
                for o in ov_segment:    # for every word in the overlap, check if the distance between the timestamp of two words is within the threshold
                    if np.absolute(o['start_time'] - s['start_time']) < timestamp_threshold:    # and o['confidence'] > min_confidence
                        if s['word'].find(o['word']) == -1 and o['word'].find(s['word']) == -1:  # if neither word is a subset of the other -> the words are completely different
                            # append o to the line
                            line = line + ' ' + (o['word'])  # add the current word in the overlap to the line, if the above conditions are true
                        elif o['word'].find(s['word']) == 0 and not len(o['word']) == len(s['word']):    # if s is subset of o (o is the longer word) and starts at index 0 of o (and the words aren't the same)
                            # replace s with o
                            line = o['word'].join(line.rsplit(s['word'], 1))    # replace the last added word s with the word o
                        else:   # for every other case, (o is subset of s, word indexes start anything other than 0) do nothing
                            pass
        else:
            line = ' '.join(s['word'] for s in segment)     # join the words in the segment into a sentence
        new_transcript.append((begin_segment, end_segment, line))
        i += 2  # jump to the next segment
    return new_transcript


def write_transcript_to_file(features_path, transcript):
    with open(os.path.join(features_path), 'w') as f:
        for t in transcript:
            line = "{0} {1} {2}\n".format(t[0], t[1], t[2])
            f.write(line)


def main(deepspeech_model, deepspeech_scorer, videos_root, features_root, segment_length_ms, overlap_length_ms, min_confidence, timestamp_threshold, videoids, idmapper):
    print("Generating transcripts for {0} videos".format(len(videoids)))
    ds = Model(deepspeech_model)
    ds.enableExternalScorer(deepspeech_scorer)

    # repeat until all movies are transcribed correctly
    done = 0
    for videoid in tqdm(videoids):
        try:
            video_rel_path = idmapper.get_filename(videoid)
        except KeyError as err:
            print("No such videoid: '{videoid}'".format(videoid=videoid))
            done += 1

        video_name = os.path.basename(video_rel_path)[:-4]
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)
        # print('features_dir',features_dir)
        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)

        features_fname_vid = "{videoid}.{extractor}.csv".format(videoid=videoid, extractor=EXTRACTOR)
        f_path_csv = os.path.join(features_dir, features_fname_vid)
        done_file_path = os.path.join(features_dir, '.done')

        v_path = os.path.join(videos_root, video_rel_path)
        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == VERSION:
            print('automatic speech recognition results missing or version did not match, generating transcript for {video_name}'.format(video_name=video_name))
            extract_wav_from_video(v_path, videoid)
            audio_path = '/tmp/' + str(videoid) + '_audio.wav'
            transcript = process_wavefile(ds, audio_path, segment_length_ms, min_confidence)
            transcript_with_overlaps = get_overlapping_segments(ds, audio_path, transcript, overlap_length_ms, min_confidence)
            new_transcript = process_transcript(transcript_with_overlaps, timestamp_threshold)
            write_transcript_to_file(f_path_csv, new_transcript)

            # create a hidden file to signal that the asr for a movie is done
            # write the current version of the script in the file
            with open(done_file_path, 'w') as d:
                d.write(VERSION)
            done += 1  # count the instances of the asr done correctly

        else:
            # do nothing if a .done-file exists and the versions in the file and the script match
            done += 1  # count the instances of the asr done correctly
            print('automatic speech recognition was already done for {video}'.format(video=video_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_path', help='path to the video to be transcribed')
    parser.add_argument('features_path', help='path to the directory where the results are saved')
    parser.add_argument('file_mappings', help='path to file mappings .tsv-file')
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument('--deepspeech_model', default='../deepspeech-0.7.4-models.pbmm', help='path to a deepspeech model, default is ../deepspeech-0.7.4-models.pbmm')
    parser.add_argument('--deepspeech_scorer', default='../deepspeech-0.7.4-models.scorer', help='path to a deepspeech scorer, default is ../deepspeech-0.7.4-models.scorer')
    parser.add_argument('--segment_length_ms', type=int, default=3000, help='lenght of the segments for which audio is transcribed, in milliseconds, default 3000')
    parser.add_argument('--overlap_length_ms', type=int, default=100, help='lenght of the overlap between two segments, in milliseconds, default ')
    parser.add_argument('--min_confidence', type=float, default=10.0, help='minimum confidence the deepspeech model has to return for a word'
                                                                           'if the confidence is lower SIL is returned instead, default is 10.0')
    parser.add_argument('--timestamp_threshold', type=int, default=50, help='maximum number of milliseconds two timestamps can be apart of each other to be considered the same, '
                                                                            'in milliseconds, default')
    args = parser.parse_args()

    idmapper = TSVIdMapper(args.file_mappings)
    videoids = args.videoids if len(args.videoids) > 0 else idmapper.get_ids()

    main(args.deepspeech_model, args.deepspeech_scorer, args.videos_path, args.features_path,
         args.segment_length_ms, args.overlap_length_ms, args.min_confidence, args.timestamp_threshold, videoids, idmapper)
