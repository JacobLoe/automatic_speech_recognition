# source $HOME/tmp/deepspeech-venv/bin/activate

from deepspeech import Model
import wave
import argparse
import numpy as np
import subprocess


def extract_wav_from_video(video_path):
    audio_path = '/tmp/audio.wav'
    command = "ffmpeg -y -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(video_path, audio_path)
    subprocess.call(command, shell=True)


def metadata_to_string(metadata):
    conf = np.abs(metadata.confidence)
    print(conf)
    if conf > 10.0:
        return ''.join(token.text for token in metadata.tokens) + " ({0})".format(conf)
    else:
        return "SIL"


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
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
            each_word["start_time "] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def process_wavefile(ds, wavefile, segment_length_ms):
    with wave.open(wavefile, 'rb') as ain:
        framerate = ain.getframerate()
        nframes = ain.getnframes()
        audio_length_ms = nframes * (1. / framerate) * 1000

        for start_ms in range(0, int(audio_length_ms), segment_length_ms):
            ain.setpos(int(start_ms / 1000. * framerate))
            chunkData = np.frombuffer(ain.readframes(int(segment_length_ms / 1000. * framerate)), np.int16)

            words = metadata_to_string(ds.sttWithMetadata(chunkData, 1).transcripts[0])

            print("{0} {1} {2}".format(start_ms, start_ms + segment_length_ms, words))


def main(deepspeech_model, deepspeech_scorer, video_path, segment_length):
    ds = Model('../deepspeech-0.7.4-models.pbmm')
    scorer = '../deepspeech-0.7.4-models.scorer'
    ds.enableExternalScorer(scorer)
    #
    # wavefile = '../audio/2830-3980-0043.wav'
    #
    segment_length_ms = 3000
    max_length_ms = 600000  # 10 min
    video_path = '../videos/unterordner/movies/Her_bluray_Szene_11_25fps.mp4'

    extract_wav_from_video(video_path)

    process_wavefile(ds, '/tmp/audio.wav', segment_length_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('deepspeech_model', help='')
    parser.add_argument('deepspeech_scorer', help='')
    parser.add_argument('video_path', help='')
    parser.add_argument('--segment_length', type=int, default=3000, help='')
    args = parser.parse_args()

    main(args.deepspeech_model, args.deepspeech_scorer, args.video_path, args.segment_length)
