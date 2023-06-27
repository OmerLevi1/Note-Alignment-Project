import mido
from mido import MidiFile
import random
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import math
from pydub import AudioSegment
from pydub.generators import WhiteNoise
# import sys
import numpy as np
import subprocess
import wave
import librosa
import librosa.display
import librosa.display
import scipy.io.wavfile as wav
from pydub import AudioSegment
import pretty_midi
import string
import torch
from glob import glob


#todo:remove?

# sys.path.insert(0,'C://Users//97252//anaconda3//envs//deep_learn//Lib')
# print(sys.path)


def cut_midi_file_mido(inp_path, output_path, new_file_name, length_sec, save=False):
    seconds_to_cut = length_sec * 420
    midifile = mido.MidiFile(inp_path)
    lasttick = seconds_to_cut  # last tick you want to keep

    for track in midifile.tracks:
        tick = 0
        keep = []
        for msg in track:
            if tick > lasttick:
                break
            keep.append(msg)
            tick += msg.time
        track.clear()
        track.extend(keep)
    if save:
        new_name = output_path + '/' + new_file_name + '.mid'
        midifile.save(new_name)
    return midifile


def cut_midi_file(inp_path, output_path, new_file_name, length_sec, save=False):
    seconds_to_cut = length_sec
    midi_data = pretty_midi.PrettyMIDI(inp_path)

    last_time = seconds_to_cut

    for instrument in midi_data.instruments:
        keep_notes = []
        for note in instrument.notes:
            if note.start < last_time:
                keep_notes.append(note)
        instrument.notes = keep_notes

    if save:
        new_file_path = os.path.join(output_path, new_file_name + ".mid")
        midi_data.write(new_file_path)

    return midi_data


def add_mistakes_to_midi_file(input_file_path, output_file_path, mistake_probability, error_range):
    mid = mido.MidiFile(input_file_path)
    noise_arr = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on':
                if random.random() < mistake_probability:
                    # Add a random value between -5 and 5 to the note value
                    curr_noise = random.randint(-error_range, error_range)
                    noise_arr.append(curr_noise)
                    new_note_value = msg.note + curr_noise

                    if new_note_value < 0:
                        new_note_value = 0
                    if new_note_value > 127:
                        new_note_value = 127
                    msg.note = new_note_value
                else:
                    noise_arr.append(0)

    mid.save(output_file_path)
    return noise_arr


def plot_chromogram(wav_input, title):
    if isinstance(wav_input, str):
        # Load the audio file from the provided path
        y, sr = librosa.load(wav_input)
    else:
        # Assume the input is a WAV file array
        y = wav_input
        sr = 44100  # Assuming a sample rate of 44100 Hz

    if len(y) < 2048:
        print("Error: Input audio signal is too short to compute the chromagram.")
        return None

    # Compute the chromagram
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048)

    # Plot the chromagram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

    return chromagram


def dtw_chromograms(file1, file2):
    # Load the audio files and compute chromagrams
    y1, sr1 = librosa.load(file1)
    chroma1 = librosa.feature.chroma_stft(y=y1, sr=sr1)

    y2, sr2 = librosa.load(file2)
    chroma2 = librosa.feature.chroma_stft(y=y2, sr=sr2)

    # Apply the DTW algorithm
    dtw_distance, dtw_path = librosa.sequence.dtw(y1, y2)
    # dtw_distance, dtw_path = librosa.sequence.dtw(chroma1.T, chroma2.T)

    # Visualize the DTW path
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma1, y_axis='chroma', x_axis='time')
    plt.plot(dtw_path[:, 0], dtw_path[:, 1], color='r', linewidth=2)
    plt.title('DTW Path')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Visualize the aligned chromagrams
    aligned_chroma1 = np.zeros_like(chroma2)
    for i, j in dtw_path:
        aligned_chroma1[:, j] = chroma1[:, i]

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(chroma1, y_axis='chroma', x_axis='time')
    plt.title('Chromagram 1')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(aligned_chroma1, y_axis='chroma', x_axis='time')
    plt.title('Aligned Chromagram 1')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def isolate_instrument_channel(midi_file_path, isolated_mid_path, channels):
    if not midi_file_path.endswith('.mid'):
        raise ValueError("The provided file is not a MIDI file.")

    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Create a new PrettyMIDI object to store the isolated instrument channel
    isolated_midi = pretty_midi.PrettyMIDI()
    for i in range(len(channels)):
        curr_inst = midi_data.instruments[channels[i]]
        print(curr_inst)
        isolated_midi.instruments.append(curr_inst)

    # Save the isolated instrument channel to a new MIDI file
    isolated_midi.write(isolated_mid_path)


def midi2wav(input_midi_path, output_wav_path):
    ## old implementation
    # dev_null = open(os.devnull, 'w')
    # timidity = 'C:\\TiMidity++-2.15.0\\timidity'
    # cmd = f"{timidity} {inp_path} -Ow -o - 2> NUL | ffmpeg -y -f wav -i - {wav_path2}"
    # # print(cmd) #show final command
    # os.system(f'cmd /k "{cmd}"') # Execute the command
    # dev_null.close()

    # dev_null = open(os.devnull, 'w')
    # timidity = 'C:\\TiMidity++-2.15.0\\timidity'
    # cmd = f"{timidity} {cut_path} -Ow -o - 2> NUL | ffmpeg -y -f wav -i - {wav_path_cut}"
    # # print(cmd) #show final command
    # os.system(f'cmd /k "{cmd}"') # Execute the command
    # dev_null.close()

    timidity = 'C:\\TiMidity++-2.15.0\\timidity'
    cmd = f"{timidity} {input_midi_path} -Ow -o - 2> NUL | ffmpeg -y -f wav -i - {output_wav_path}"
    subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)

    # Load the WAV file
    rate, data = wav.read(output_wav_path)
    data = data.astype(np.float32) / 32767.0  # Normalize the audio data

    return data

#todo:remove?
def load_wav_file(wav_file_path):
    try:
        import scipy.io.wavfile as wav
    except ImportError:
        raise ImportError("The 'scipy' library is required to load WAV files. "
                          "You can install it using 'pip install scipy'.")

    rate, data = wav.read(wav_file_path)
    return np.array(data, dtype=np.float32) / 32767.0  # Normalize the audio data


def print_existing_tracks(midi_file_path):
    if not midi_file_path.endswith('.mid'):
        raise ValueError("The provided file is not a MIDI file.")

    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    num_tracks = len(midi_data.instruments)

    print("Existing Tracks:")
    for i, instrument in enumerate(midi_data.instruments):
        print(f"Track {i+1}: {instrument.name if instrument.name else 'Unnamed Track'}")

    print(f"Total Tracks: {num_tracks}")


def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]


def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result


def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]


def track2seq(track):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result


def mid2array(path, min_msg_pct=0.1, plot=False):
    mid = mido.MidiFile(path)
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    result_array = all_arys[min(ends): max(ends)]
    if plot:
        plt.plot(range(result_array.shape[0]), np.multiply(np.where(result_array > 0, 1, 0), range(1, 89)), marker='.',
                 markersize=1, linestyle='')
        plt.title("array of midi file")
        plt.show()
    return result_array


def mid_arr_compressor(input_array, output_path):
    # Create a list to store the notes that were played at each time sample.
    notes_played = []

    # Iterate over the rows of the input array.
    for row in input_array:
        # Check if any of the notes in the row are not zero.
        if any(row):
            # If so, add the list of notes to the notes_played list.
            notes_played.append([i for i, x in enumerate(row) if x != 0])
        else:
            notes_played.append(0)

    # Convert the notes_played list to a NumPy array.
    output_array = np.array(notes_played)

    # Flatten the array while keeping the integers separate
    flattened_arr = [x for sublist in output_array for x in (sublist if isinstance(sublist, list) else [sublist])]

    # Convert the flattened list to a NumPy array
    output_array = np.array(flattened_arr)

    #
    # # Convert the input array to a list.
    # notes_list = output_array.tolist()
    #
    # # Remove the brackets from the output list.
    # for i in range(len(notes_list)):
    #     if isinstance(notes_list[i], list):
    #         notes_list[i] = notes_list[i][0]
    #
    # output_array = np.array(notes_list)

    # # Save the output array to a text file.
    # with open(output_array, 'w') as f:
    #     for note in output_array:
    #         f.write(str(note) + '\n')

    np.savetxt(output_path, flattened_arr, fmt='%d')

    return output_array


# TODO: doesnt work
def ctc_alignment(array1, array2):
#   """
#   Performs CTC alignment on two output arrays from the mid_arr_compressor function using CTC.
#
#   Args:
#     array1: The first output array from the mid_arr_compressor function.
#     array2: The second output array from the mid_arr_compressor function.
#
#   Returns:
#     A list of pairs of notes, where each pair represents a note that was played in both arrays.
#   """
#
#   # Convert the input arrays to NumPy arrays.
#   array1 = np.array(array1)
#   array2 = np.array(array2)
#
#   # Create a list to store the aligned notes.
#   aligned_notes = []
#
#   # Convert the input arrays to tensors.
#   array1_tensor = torch.tensor(array1)
#   array2_tensor = torch.tensor(array2)
#
#   # CTC alignment.
#   alignment = ctc.ctc_align(array1_tensor, array2_tensor, blank=0)
#
#   # Convert the alignment to a list of pairs of notes.
#   for i in range(len(alignment)):
#     for j in range(len(alignment[i])):
#       aligned_notes.append((alignment[i][j][0], alignment[i][j][1]))
#
#   # Return the list of aligned notes.
#   return aligned_notes
    return True


#todo:remove?
def convert_midi_to_txt(midi_file):
  """Converts a MIDI file with a single track to a text file that indicates which notes were played when.

  Args:
    midi_file: The path to the MIDI file.

  Returns:
    A text file that contains a list of notes, separated by spaces. Each note is represented by a tuple of the form (note, octave).
  """

  # Open the MIDI file.
  with mido.MidiFile(midi_file) as midi:

    # Get the track.
    track = midi.tracks[1]

    # Create a list to store the notes.
    notes = []

    # Iterate over the events in the track.
    for event in track:

      # If the event is a note on event, add the note to the list.
      if event.type == 'note_on':
        notes.append((event.note, event.channel))

      # If the event is a note off event, remove the note from the list.
      elif event.velocity == 0:
        notes.append((0, event.channel))

    # Convert the list of notes to a text file.
    txt_file = open("notes.txt", "w")
    for note in notes:
      txt_file.write("%d %d " % note)
    txt_file.close()

  return "notes.txt"


def get_non_drum_channels(input_data):
    if isinstance(input_data, str):
        mid = pretty_midi.PrettyMIDI(input_data)
    elif isinstance(input_data, mido.MidiFile):
        mid = input_data
    else:
        raise ValueError("Invalid input data. Please provide a file path or a mido.MidiFile object.")

    non_drum_channels = []
    for i, instrument in enumerate(mid.instruments):
        if not instrument.is_drum:
            non_drum_channels.append(i)
    return non_drum_channels


def create_demos(input_path, cut_dir_path, isolated_clean_path, isolated_err_path, length_sec, files_num, mistake_probability, error_range):
    # Get a list of MIDI files in the input path
    midi_files = glob(os.path.join(input_path, "*.mid"))
    # Take only the first "files_num" files
    midi_files = midi_files[:files_num]

    for midi_file in midi_files:
        # Extract the file name without the extension
        file_name = os.path.splitext(os.path.basename(midi_file))[0]

        # Cut the MIDI file
        cur_cut_path = cut_dir_path + "/" + file_name + ".mid"
        cut_mid = cut_midi_file(midi_file, cut_dir_path, "/" + file_name, length_sec, save=True)

        # Find non-drum tracks
        noDrumTracks = get_non_drum_channels(cur_cut_path)

        # Iterate over the non-drum tracks
        for index in noDrumTracks:
            isolated_clean_file_name = f"{file_name}_track_{index}_clean.mid"
            isolated_clean_file_path = isolated_clean_path + "/" + isolated_clean_file_name
            isolated_err_file_name = f"{file_name}_track_{index}_err.mid"
            isolated_err_file_path = isolated_err_path + "/" + isolated_err_file_name

            # isolate and save current non-drum track
            isolate_instrument_channel(cur_cut_path, isolated_clean_file_path, [index])


            # Add mistakes to the isolated track
            noise_arr = add_mistakes_to_midi_file(isolated_clean_file_path, isolated_err_file_path, mistake_probability, error_range)

#------------------------------------ MAIN ------------------------------------#
if __name__ == '__main__':

    # --------- paths --------- #
    lmd0 = "C:/Users/97252/PycharmProjects/MusicProject/lmd_full/0"
    mid_cut = "C:/Users/97252/PycharmProjects/MusicProject/mid_cut"
    isolated_err_dir = "C:/Users/97252/PycharmProjects/MusicProject/isolated_err"
    isolated_clean_dir = "C:/Users/97252/PycharmProjects/MusicProject/isolated_clean"

    dir_path = "C:/Users/97252/PycharmProjects/MusicProject"
    inp_path = "C:/Users/97252/PycharmProjects/MusicProject/demo.mid"
    cut_path = "C:/Users/97252/PycharmProjects/MusicProject/demo_cut.mid"
    cut_for_err_path = "C:/Users/97252/PycharmProjects/MusicProject/demo_cut_for_err.mid"
    err_path = "C:/Users/97252/PycharmProjects/MusicProject/demo_cut_err.mid"
    wav_path = "C:/Users/97252/PycharmProjects/MusicProject/out.wav"
    wav_path2 = "C:/Users/97252/PycharmProjects/MusicProject/out2.wav"
    wav_path_cut = "C:/Users/97252/PycharmProjects/MusicProject/out_cut.wav"
    wav_path_err = "C:/Users/97252/PycharmProjects/MusicProject/out_err.wav"
    wav_path_isolated_err = "C:/Users/97252/PycharmProjects/MusicProject/isolated_err.wav"
    mid_path_isolated_err = "C:/Users/97252/PycharmProjects/MusicProject/isolated_err.mid"
    wav_path3 = "C:/Users/97252/PycharmProjects/MusicProject/testoosh.wav"
    wav_path4 = "C:/Users/97252/PycharmProjects/MusicProject/delete_me_please.wav"
    wav_path_isolated = "C:/Users/97252/PycharmProjects/MusicProject/isolated_track.wav"
    mid_path_isolated = "C:/Users/97252/PycharmProjects/MusicProject/isolated_track.mid"
    mid_no_drums = "C:/Users/97252/PycharmProjects/MusicProject/no_drums.mid"
    trash = "C:/Users/97252/PycharmProjects/MusicProject/trash.txt"
    # wav_path2 = "C:\\Users\\97252\\PycharmProjects\\MusicProject\\a.wav"
    # sf2_path = "C:/Users/97252/PycharmProjects/MusicProject/gm.sf2"
    length_sec = 15  # length of new cut file
    # cut_file_name = 'demo_cut'
    cut_file_name = 'demo_cut_pm'

    # --------- cut midi file --------- #
    cut_midi_file(inp_path, dir_path, cut_file_name,  length_sec, True)  # take first part of file

    # --------- isolate tracks --------- #
    ## print_existing_tracks(cut_path) # show all tracks
    # tracks = [3] # choose tracks
    # isolated_channel_midi = isolate_instrument_channel(cut_path, mid_path_isolated, tracks) # isolate midi to midi
    # isolated_channel_wav = midi2wav(mid_path_isolated, wav_path_isolated) # convert midi to wav
    ## isolated_chrom = plot_chromogram(wav_path_isolated, "Chromagram of Isolated") # chromagram of isolated


    # --------- save txt of midi (single channel) --------- #
    # mid_arr_orig = mid2array(mid_path_isolated, plot=False)  # get isolated track as 2d array
    # compressed_arr_orig = mid_arr_compressor(mid_arr_orig, trash)  # get isolated track in compressed 1d array
    # mid_arr_err = mid2array(mid_path_isolated_err, plot=False)  # same with error of isolated
    # compressed_arr_err = mid_arr_compressor(mid_arr_err, trash)
    # ctc = ctc_alignment(compressed_arr_orig, compressed_arr_err)  # perform CTC

    # --------- get non-drum tracks --------- #
    # noDrumTracks = get_non_drum_channels(cut_path)  # get a list of indices of non drum tracks
    # drumless_mid = isolate_instrument_channel(cut_path, mid_no_drums, noDrumTracks)  # isolate midi to midi

    # --------- add errors to midi --------- #
    # mistake_probability = 0.1  # 10% chance of introducing a mistake
    # error_range = 2
    # noise_arr = add_mistakes_to_midi_file(mid_path_isolated, err_path, mistake_probability, error_range)

    # --------- add mistakes to isolated midi track and convert to wav --------- #
    # mistake_iso_err_wav = add_mistakes_to_midi_file(mid_path_isolated, mid_path_isolated_err, mistake_probability, error_range)
    # isolated_channel_err_wav = midi2wav(mid_path_isolated_err, wav_path_isolated_err)

    # --------- perform DTW and plot results --------- #
    # dtw_chromograms(wav_path_isolated, wav_path_isolated_err)


    # --------- create dataset--------- #
    input_path = lmd0
    cut_dir_path = mid_cut
    isolated_clean_path = isolated_clean_dir
    isolated_err_path = isolated_err_dir
    length_sec = 15
    files_num = 5
    mistake_probability = 0.1
    error_range = 2
    # create_demos(input_path, cut_dir_path, isolated_clean_path, isolated_err_path, length_sec, files_num, mistake_probability, error_range)


    print('ahlan olam')

