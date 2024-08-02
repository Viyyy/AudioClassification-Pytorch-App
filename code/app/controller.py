
import os
import json
import torchaudio

from .common.colors import get_color_dict
from .transfomers import get_melspec_transformer
from .plt_helper import draw_pie, plot_spectrogram, plt2ndarray
from .predictor import get_model_names, get_model_keys, get_predictor, get_labels_info

RESULT_DATA_THRESHOLD = float(os.getenv('RESULT_DATA_THRESHOLD', 0.05))
RESULT_DATA_TOTAL_THRESHOLD = float(os.getenv('RESULT_DATA_TOTAL_THRESHOLD', 0.05))

from pydantic import BaseModel, Field
from typing import Union, Any

class PredictDataController(BaseModel):
    signal: Any = Field(default=None, description="当前的音频信号")
    sr: Union[int, None] = Field(default=None, description="当前的音频采样率")
    predictor: Any = Field(default=None, description="当前选择的模型")
    labels_info: Union[dict, None] = Field(default=None, description="当前模型的标签信息")
    result: Union[list, None] = Field(default=None, description="当前识别结果")
    result_total: Union[list, None] = Field(default=None, description="当前大类的识别结果")
    color_dict: Union[dict, None] = Field(default=None, description="当前颜色配色")
    color_dict_total: Union[dict, None] = Field(default=None, description="当前大类的颜色配色")

    @property
    def total_labels(self)->Union[list, None]:
        if self.labels_info is not None:
            return list(set(self.labels_info.values()))
        else:
            return None
        
    supported_models:list = Field(default_factory=get_model_names, description="模型名称列表") 

    kw_list:list = Field(default_factory=get_model_keys, description="模型关键词列表")

    mel_spec_transformer:Any = Field(default=get_melspec_transformer(n_fft=1024, hop_length=512, n_mels=64, sample_rate=16000), description="Mel-Spectrogram 转换器")

    dpi:int = Field(default=200, description="图片分辨率")
    
    def load_audio(self, audio_file):
        try:
            if audio_file is None:
                self.signal = None
                self.sr = None
                return None, None, None
            signal, sr = torchaudio.load(audio_file)
            self.signal = signal
            self.sr = sr
            return os.path.basename(audio_file), self.signal.shape, self.sr
        except Exception as e:
            raise e
        finally:
            self.clear_result()
            self.clear_result_total()

    def get_model(self, model_name, model_key):
        try:
            self.predictor = get_predictor(model_name, model_key)
            self.labels_info = get_labels_info(model_name, model_key)
            if self.predictor is None:
                return "没有这个模型哦，换一个吧"
            else:
                return self.predictor.configs
        except Exception as e:
            raise e
        finally:
            self.clear_result()
            self.clear_result_total()

    def get_labels(self, color_seed):
        if self.predictor is None:
            return None
        color_dict = get_color_dict(
            self.predictor.class_labels, random_state=color_seed
        )
        self.color_dict = {k: v["hex"] for k, v in color_dict.items()}

        if self.labels_info is not None:

            color_dict_total = get_color_dict(self.total_labels, random_state=color_seed)
            self.color_dict_total = {
                k: v["hex"] for k, v in color_dict_total.items()
            }

        return json.dumps(
            {
                "labels": self.color_dict,
                "total_labels": self.color_dict_total,
            },
            ensure_ascii=False,
        )

    def predict(self):
        predictor = self.predictor

        if predictor is None:
            return None

        self.result = predictor.predict(self.signal.numpy()[0], sample_rate=self.sr)
        
        if self.labels_info is not None:
            result_total = {c: 0 for c in self.total_labels}
            for label, prob in self.result:
                # 找到大类
                category = self.labels_info.get(label, None)
                if category is not None:
                    result_total[category] += prob

            self.result_total = [tuple([k, v]) for k, v in result_total.items()]

        return {
            "result": self.result,
            "total": self.result_total,
        }

    def draw_result_img(self, fontsize=10, thresold=RESULT_DATA_THRESHOLD):
        if self.result is None:
            return None

        fig = draw_pie([self.result], self.color_dict, fontsize=fontsize, threshold=thresold)

        img = plt2ndarray(fig, dpi=self.dpi)

        return img
    
    def draw_result_total_img(self, fontsize=12, thresold=RESULT_DATA_TOTAL_THRESHOLD):
        if self.result_total is None:
            return None
        fig_total = draw_pie(
            [self.result_total], self.color_dict_total, fontsize=fontsize, threshold=thresold
        )

        img_total = plt2ndarray(fig_total, dpi=self.dpi)

        return img_total

    def draw_spectrogram_img(self):
        if self.signal is None:
            return None
        
        feature = self.mel_spec_transformer(self.signal[0])
        fig = plot_spectrogram(feature, title="Mel-Spectrogram")
        img = plt2ndarray(fig, dpi=self.dpi)
        return img

    def clear_result(self)->None:
        self.result = None
        return None
    
    def clear_result_total(self)->None:
        self.result_total = None
        return None