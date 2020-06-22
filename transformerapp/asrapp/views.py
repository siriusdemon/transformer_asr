from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View
from asrapp.asr import Recognition
import os
import json
import time
# Create your views here.
def hello(request):
    return HttpResponse('Hello World!')
class SpeechRecognitionView(View):
    def post(self, request):
        audio_file = request.FILES.get('audio', None) #音频文件
        if not audio_file:
            return HttpResponse({"upload audio error！"})
        filecontent = audio_file.read()
        temp_time = str(time.time())
        filename=''.join(audio_file.name)
        if len(filename.split('.')[-1])==4:
            filename=filename[:-1]
        #print(filename)
        filename = str(filename[:-4])+temp_time[-6:] + str(filename[-4:])
        #print([filename])
        with open(filename, 'wb') as f:
            f.write(filecontent)
        response_data = Recognition(filename) #语音识别
        response_data = json.dumps(response_data, ensure_ascii=False)
        os.remove(filename)
        return HttpResponse(response_data) # 返回json格式
