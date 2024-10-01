# Llamafactory_FT_Qwen
基于Llamaindex微调qwen2.5-7b

## 1.数据集准备

下载需要处理的文本数据，最好转置为txt文件，然后使用bert模型进行chunk切分

切分后使用百炼的API调用大模型进行数据集构建

https://www.cnblogs.com/chentiao/p/17386131.html
https://blog.csdn.net/yierbubu1212/article/details/142635578?spm=1001.2014.3001.5502

## 2.服务器搭建

由于算力限制选择阿里云服务器

https://free.aliyun.com/?spm=a2c4g.11174283.0.0.46a0527f6LNsEe&productCode=learn

https://help.aliyun.com/document_detail/2329850.html?spm=a2c4g.2261126.0.0.f3be1d2ddJscIy

## 3.Qwen2.5-7b

`!git lfs install` 记得下载lfs不然无法下载完整的模型文件

`!git clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git`

## 4.Llamafactory

拉取环境

`!git clone https://github.com/hiyouga/LLaMA-Factory.git`

cd到Llamafactory目录下,加载依赖项
`!pip install -e .[torch,metrics]`

### 4.1加载数据

将转换好的.json数据放入Llamafactory中的data文件中，并修改配置文件data_set.json

``` python
  "train": {
    "file_name": "your_data_name.json"
  },
  "test": {
    "file_name": "your_data_name.json"
  }
```

### 4.2trian

cd到Llamafactory目录下 
`!llamafactory-cli webui` 启动webui界面

![alt text](./images/image.png)
下拉找到训练就可以开始微调训练
![alt text](./images/image-0.png)

---

- 参数配置
  
  train中的可选参数很多，这里先搁置不进行一一介绍，感兴趣的小伙伴可以自己先进行了解
  
### 4.3Evaluate & Predict

![alt text](./images/image-1.png)


- 评价指标解释
  - predict_bleu-4 (30.835%):
    BLEU-4分数（BiLingual Evaluation Understudy）BLEU是衡量机器翻译或文本生成模型输出与参考答案相似度的常用指标。bleu-4表示计算了4-gram的匹配率，数值越高表示预测的文本质量越高。

  - predict_model_preparation_time (0.004秒):
    模型在进行预测前的准备时间，通常指模型加载和初始化的时间，单位是秒。数值越小表示准备时间越短。
  - predict_rouge-1 (52.03%):
    ROUGE-1是文本生成评价指标之一，它衡量生成文本与参考文本之间的1-gram重叠程度。rouge-1的数值越高，代表模型预测的词汇与参考答案越接近。
  - predict_rouge-2 (28.15%):
    ROUGE-2同样是ROUGE系列指标之一，衡量2-gram的重叠率。和ROUGE-1一样，数值越高，代表模型输出的文本与目标文本越接近。
  - predict_rouge-l (38.01%):
    ROUGE-L使用最长公共子序列（LCS）来衡量模型生成文本和参考文本之间的相似度。该指标更关注文本的顺序结构，数值越高代表输出文本在顺序上与参考答案更相似。
  - predict_runtime (1255.0994秒):
    这是模型进行预测所花费的总时间，单位为秒。该值通常会随模型复杂度、硬件配置以及数据集大小而变化。
  - predict_samples_per_second (0.17 样本/秒):
    这是模型每秒能够处理的样本数。值越高表示模型的处理效率越高。
  - predict_steps_per_second (0.085 步骤/秒):
    这是模型每秒执行的步骤（steps），每个步骤通常包括一次前向传播和梯度更新。数值越高，表示模型在预测过程中的计算效率越高。
### 4.4Export
（chat部分就是一个聊天界面，可以在线测试微调后的效果）
Export是将我们微调的模型进行导出，我的理解是(将训练好的lora部分和原本的模型进行合并)

这部分只用设置输出路径就可以进行导出了

## 5部署

## 5.1 Ollama部署

在服务器上部署

首先拉取ollma文件

`modelscope download --model=modelscope/ollama-linux --local_dir ./ollama-linux --revision v0.3.12`

新建创建 Modelfile 文件，写入

`FROM ./your_model_path`

- 在ollama文件中创建模型

`ollama create mymodel -f Modelfile`

- Llama.cpp

    如果在ollama创建模型文件的时候遇见 `Models based on 'Qwen2ForCausalLM' are not yet supported `的问题，可使用llama.cpp,导出gguf格式的文件再进行部署推理

    首先拉取llama.cpp文件

    `!git clone https://github.com/ggerganov/llama.cpp.git`

    ```python
    # 需要安装相关的库
    cd llama.cpp
    pip install -r requirements.txt
    # 验证环境
    python convert_hf_to_gguf.py -h

    # 使用脚本进行模型转换，可以选择量化方式
    python convert_hf_to_gguf.py ../yourmodelpath --outfile out_file_name.gguf --outtype f16
    ```
得到.gguf文件
然后重新使用`ollama create mymodel -f Modelfile`创建模型

- 最后使用ollama推理
`ollama run mymodel`

此时推理出现 输出不停止的情况？





