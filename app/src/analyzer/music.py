import os

import librosa
import madmom
import numpy as np
import json

from musicai_sdk import MusicAiClient

from utils.util import fetch_json, check_include, normalize, hz_to_midi


def needleman_wunsch(seq1, seq2, match=2, mismatch=-1, gap=-1):
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        score[i][0] = gap * i
    for j in range(m + 1):
        score[0][j] = gap * j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = score[i - 1][j - 1] + (match if seq1[i - 1]['word'] == seq2[j - 1]['word'] else mismatch)
            delete = score[i - 1][j] + gap
            insert = score[i][j - 1] + gap
            score[i][j] = max(match_score, delete, insert)

    align1, align2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        current_score = score[i][j]
        if i > 0 and j > 0 and (current_score == score[i - 1][j - 1] + (
        match if seq1[i - 1]['word'] == seq2[j - 1]['word'] else mismatch)):
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (current_score == score[i - 1][j] + gap):
            align1.append(seq1[i - 1])
            align2.append({'word': '-', 'meta': '-'})
            i -= 1
        else:
            align1.append({'word': '-', 'meta': '-'})
            align2.append(seq2[j - 1])
            j -= 1

    return align1[::-1], align2[::-1]


class MusicAnalyzer:
    def __init__(self, url, original_lyrics):
        self.client = MusicAiClient(api_key=os.getenv('MUSICAI_API_KEY'))
        print(f'Application Info: {self.client.get_application_info()}')

        self.url = url
        self.original_lyrics = original_lyrics
        self.original_dict = []
        if self.original_lyrics != '':
            self._set_original_dict()
            self.workflow = 'synthesizer'
        else:
            self.workflow = 'synthesizer_instrument'

        self.bpm = None
        self.lyrics = []
        self.metadata = None
        self.vocal_pitch = None
        self.aligned_lyrics = []
        self.beat_amp = []
        self.pitches = []

    def _set_original_dict(self):
        # remove all strings in []
        lyrics = [lyric for lyric in self.original_lyrics.split('\n') if not lyric.startswith('[') and not lyric.endswith(']')]
        for i, phase in enumerate(lyrics):
            for word in phase.split():
                self.original_dict.append({'word': word, 'phase': i})

    def analyze(self):
        file_url = self.client.upload_file(file_path=self.url)
        workflow_params = {
            'inputUrl': file_url
        }
        create_job_info = self.client.create_job(
            job_name='music_lyrics',
            workflow_id=self.workflow,
            params=workflow_params)
        job_id = create_job_info['id']
        print(job_id)

        job_info = self.client.wait_for_job_completion(job_id)

        status = job_info['status']
        results = job_info['result']

        if status != 'SUCCEEDED':
            print(f'MusicAI Error occurred status response is {status}')
        else:
            self.bpm = results['BPM']
            self.metadata = fetch_json(results['Music metadata'])
            if len(self.original_dict) > 0:
                lyrics = fetch_json(results['Lyrics'])
                for phase in lyrics:
                    for word in phase['words']:
                        self.lyrics.append(word)
                # self.chords_map = fetch_json(results['Chords map'])
                self.vocal_pitch = fetch_json(results['Vocal pitch'])

                self._align_lyrics()
                self._align_music_features()
            self._get_beat_amplitudes()
            self._get_all_pitches()

    def _align_lyrics(self):
        org_align, pred_align = needleman_wunsch(self.original_dict, self.lyrics)
        pred_bag = []
        org_bag = []
        for org, pred in zip(org_align, pred_align):
            if org['word'] == '-':
                if pred['word'] != '-':
                    pred_bag.append(pred)
            else:
                if pred['word'] == '-':
                    if len(pred_bag) > 0:
                        start_time = pred_bag[0]['start']
                        end_time = pred_bag[-1]['end']
                    else:  # 뒤에 나올 단어 아니면 앞에 단어와 합칠 것
                        org_bag.append(org)
                        continue
                else:
                    if len(pred_bag) > 0:
                        start_time = pred_bag[0]['start']
                    else:
                        start_time = pred['start']
                    end_time = pred['end']

                if len(org_bag) > 0:
                    for missed_org in org_bag:
                        if missed_org['phase'] == org['phase']:
                            org['word'] = missed_org['word'] + org['word']
                        else:
                            self.aligned_lyrics[-1]['word'] = self.aligned_lyrics[-1]['word'] + missed_org['word']
                    org_bag = []
                self.aligned_lyrics.append(
                    {'word': org['word'],
                     'start': start_time,
                     'end': end_time,
                     'phase': org['phase'],
                     'pitch': [],
                     # 'chords': []
                     })

                pred_bag = []
        print('Lyrics alignment done')

    def _align_music_features(self):
        # for chord in self.chords_map:
        #     chord_start = chord['start']
        #     chord_end = chord['end']
        #     for lyric in self.aligned_lyrics:
        #         if check_include(lyric['start'], lyric['end'], chord_start, chord_end):
        #             lyric['chords'].append({
        #                 'chord': chord['chord_majmin'],
        #                 'start': chord_start,
        #                 'end': chord_end
        #             })

        for pitch in self.vocal_pitch:
            pitch_start = pitch['start']
            pitch_end = pitch['end']
            for lyric in self.aligned_lyrics:
                if check_include(lyric['start'], lyric['end'], pitch_start, pitch_end):
                    lyric['pitch'].append({
                        'note_name': pitch['note_name'],
                        'midi_note': pitch['midi_note'],
                        'start': pitch_start,
                        'end': pitch_end
                    })
        print(f'Align music feature done')

    def get_final_format(self):
        return {
            'BPM': self.bpm,
            'Instruments': self.metadata['instrumentTags'],
            'Emotions': self.metadata['moodTags'],
            'Lyrics': self.aligned_lyrics,
            'Beat_amplitude': self.beat_amp,
            'Pitch': self.pitches,
        }

    def _get_beat_amplitudes(self):
        y, sr = librosa.load(self.url, sr=None ,dtype='float64')
        proc = madmom.features.beats.RNNBeatProcessor()
        act = proc(self.url)
        beats = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(act)
        # print(beats)
        beat_samples = librosa.time_to_samples(beats, sr=sr)
        # print(beat_samples)
        beat_samples = beat_samples[beat_samples < len(y)]
        beat_amplitude = y[beat_samples]
        # print(beat_amplitude)

        normalized_amp = normalize(beat_amplitude)
        for beat, amp in zip(beats, normalized_amp):
            self.beat_amp.append({'time': float(beat), 'amplitude': float(amp)})

        return y, sr, beats, normalized_amp

    def _get_all_pitches(self):
        y, sr = librosa.load(self.url, dtype='float64')
        harmonic, percussive = librosa.effects.hpss(y)
        pitches, magnitudes = librosa.core.piptrack(y=harmonic, sr=sr)

        pitch_array = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 0:
                pitch_array.append(pitch)
            else:
                pitch_array.append(np.nan)

        midi_pitches = [hz_to_midi(p) if not np.isnan(p) else 0 for p in pitch_array]
        times = librosa.times_like(np.array(midi_pitches), sr=sr)
        for t, p in zip(times, midi_pitches):
            self.pitches.append({'time': t, 'pitch': p})


if __name__ == "__main__":
    import dotenv

    # original_json = '몽구리 귀여워 몽멍멍뭉\n히히 웃으며 뛰어놀지\n해맑은 얼굴 몽구리의 세상\n모두가 웃는 축제의 날\n몽구리 잡아 무야지 호로로로록\n모두 다 함께 즐거운 시간\n몽구리 웃음소리\n행복이 가득 몽구리와 함께 꿈을 꾸어봐\n몽구리의 하루 밝고 빛나는\n모두의 마음을 따뜻하게\n몽구리 잡아무야지 호로로로록\n모두 다 함께 즐거운 시간'
    original_json = ''
    dotenv.load_dotenv()
    la = MusicAnalyzer('../data/music.wav', original_json)
    la.analyze()
    result = la.get_final_format()
    # save result as json
    with open('result_instrument.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

