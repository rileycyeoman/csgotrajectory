from transformers import EncoderDecoderModel
import torch
import lzma, json, awpy
import os
import pandas as pd
from tqdm import tqdm
import pickle
import os
import lzma
import json
import pandas as pd
from tqdm import tqdm

class DemoParser:
    def __init__(self, path):
        self.path = path
        self.demo_files = self._get_demo_files()
        self.parsed_demos = []
        self.map_list = ['de_nuke', 'de_inferno', 'de_vertigo', 'de_dust2', 'de_mirage', 'de_overpass', 'de_ancient']

    def _get_demo_files(self):
        return [self.path + f for f in os.listdir(self.path)]
    
    def read_parsed_demo(self, filename):
        with lzma.LZMAFile(filename, "rb") as f:
            return json.load(f)

    def create_frame_row(self, frame, demoID, mapName, roundNum):
        frame_data = [demoID, mapName, roundNum, frame["seconds"]]
        # Team specific info (CT)
        for p in frame["ct"]["players"]:
            player_info = [p["isAlive"], p["x"], p["y"], p["z"], p["velocityX"], p["velocityY"], p["velocityZ"], p["viewX"], p["viewY"]]
            frame_data.extend(player_info)
        # Team specific info (T)
        for p in frame["t"]["players"]:
            player_info = [p["isAlive"], p["x"], p["y"], p["z"], p["velocityX"], p["velocityY"], p["velocityZ"], p["viewX"], p["viewY"]]
            frame_data.extend(player_info)
        return frame_data

    def parse_demos(self):
        for f in tqdm(self.demo_files):
            demo = self.read_parsed_demo(f)
            if demo["mapName"] in self.map_list:
                parsed_frames_df = []
                for r in demo["gameRounds"]:
                    parsed_frames_df_round = []
                    ct_win = 1

                    if r["roundEndReason"] in ["CTWin", "TargetSaved", "BombDefused", "TargetBombed", "TerroristsWin"]:
                        if r["roundEndReason"] not in ["CTWin", "TargetSaved", "BombDefused"]:
                            ct_win = 0
                        for fr in r["frames"]:
                            if (fr["ct"]["players"] is not None) & (fr["t"]["players"] is not None) & (fr["clockTime"] != "00:00") & (fr["t"]["alivePlayers"] >= 0) & (fr["ct"]["alivePlayers"] >= 1):
                                if (len(fr["ct"]["players"]) == 5) & (len(fr["t"]["players"]) == 5):
                                    frame_row = self.create_frame_row(fr, demo["demoId"], demo["mapName"], r["roundNum"])
                                    frame_row.append(ct_win)
                                    parsed_frames_df_round.append(frame_row)
                    if len(parsed_frames_df_round) > 0:
                        self.parsed_demos.extend(parsed_frames_df_round)

    def save_to_pickle(self, filename='lan_frames_df.pkl'):
        lan_frames_df = pd.DataFrame(self.parsed_demos)
        lan_frames_df.to_pickle(filename)

    def load_from_pickle(self, filename='lan_frames_df.pkl'):
        return pd.read_pickle(filename)

    def demo_stats(self, demoID, roundNum):
        lan_frames_df = pd.DataFrame(self.parsed_demos)
        test = lan_frames_df[lan_frames_df[0] == demoID]
        test = test[test[2] == roundNum]
        return test

    def dataframe_shape(self):
        lan_frames_df = pd.DataFrame(self.parsed_demos)
        return lan_frames_df.shape

    def dataframe_head(self):
        lan_frames_df = pd.DataFrame(self.parsed_demos)
        return lan_frames_df.head()

    
    