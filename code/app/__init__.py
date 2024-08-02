# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-07-31 09:41:24
# Description: 音频识别应用
"""
按需加载模型，然后输出模型参数，输出标签类别: config & label_list

根据用户输入的音频，识别其类别，绘图，返回结果列表，并显示在界面上。
"""

import gradio as gr

from .controller import PredictDataController    

def demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        data_controller = PredictDataController() # 实例化数据控制器，相当于后端
        demo.title = "Audio Classification"
        # 第一行
        with gr.Row():
            with gr.Column(variant="panel"):
                audio = gr.Audio(label="Step1 上传音频文件", sources=["upload"], type="filepath")
                audio_name = gr.Text(
                    label="当前音频文件",
                    value=None,
                    lines=1,
                )
                with gr.Row():
                    sr_label = gr.Text(value=None, label="采样率")  # 采样率
                    shape_label = gr.Text(value=None, label="音频形状")  # 音频形状

                spectrogram = gr.Image(
                    value=None, label="频谱图", format="png"
                )  # 频谱图

            with gr.Column():
                with gr.Row():
                    model_name = gr.Dropdown(label="Step2 选择模型", choices=data_controller.supported_models)
                    model_key = gr.Dropdown(label="Step3 选择模型关键字", choices=data_controller.kw_list)
                labels = gr.TextArea(
                    value=None,
                    label="标签-颜色",
                    show_copy_button=True,
                    text_align="left",
                    placeholder="请先选择模型吧",
                    lines=7,
                )
                color_seed = gr.Number(
                    value=42,
                    label="Step4 随机颜色种子", info="用于生成随机颜色"
                )
                result = gr.TextArea(value=None, label="识别结果", lines=5)  # 识别结果
                # 识别按钮
                predict_btn = gr.Button("Step5 识别")

        with gr.Row(variant="panel"):
            result_img = gr.Image(
                value=None, label="饼状图", format="png"
            )  # 识别结果图片
            result_img_total = gr.Image(
                value=None, label="大类-饼状图", format="png"
            )  # 识别结果图片

        model_info = gr.TextArea(
            value=None,
            label="模型配置",
            show_copy_button=True,
            text_align="left",
            placeholder="请先选择模型吧",
            lines=5,
        )

        # region 绑定组件事件
        ''' 1.音频改变时，重新加载音频，更新音频信息；绘制频谱图；清除预测结果'''
        audio.change(
            data_controller.load_audio, inputs=audio, outputs=[audio_name, shape_label, sr_label]
        )
        audio_name.change(data_controller.draw_spectrogram_img, outputs=spectrogram)
        audio.change(lambda x: (None, None, None), outputs=[result, result_img, result_img_total])

        ''' 2.模型名字/关键字改变时，重新加载模型；更新模型信息，标签信息；清除预测结果'''
        model_key.change(
            data_controller.get_model, inputs=[model_name, model_key], outputs=[model_info]
        )
        model_name.change(
            data_controller.get_model, inputs=[model_name, model_key], outputs=[model_info]
        )
        model_info.change(data_controller.get_labels, inputs=color_seed, outputs=labels)
        model_info.change(lambda x: (None, None, None), outputs=[result, result_img, result_img_total])

        ''' 3.随机颜色种子改变时，清除预测结果图片；修改标签颜色；重新绘制结果图片'''
        color_seed.change(lambda x: (None, None), outputs=[result_img, result_img_total])
        color_seed.change(data_controller.get_labels, inputs=color_seed, outputs=labels)
        color_seed.change(data_controller.draw_result_img, outputs=[result_img])
        color_seed.change(data_controller.draw_result_total_img, outputs=[result_img_total])

        ''' 4.预测结果修改时，重新绘制图片'''
        result.change(data_controller.draw_result_img, outputs=[result_img])
        result.change(data_controller.draw_result_total_img, outputs=[result_img_total])
        # endregion

        ''' 5.点击预测按钮时，进行预测，更新预测结果'''
        predict_btn.click(data_controller.predict, outputs=[result])

    return demo
