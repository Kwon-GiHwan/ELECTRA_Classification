## Fine Tuning for Making ELECTRA Classification Model

HuggingFace Transformer 모듈의 KoELECTRA 모델을 활용하여 구성한 Classification 모델 학습코드입니다.

기존 HuggingFace에서는 Classification Task를 위한 [ElectraForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/electra#transformers.ElectraForSequenceClassification)클래스를 지원하며 다음과 같이 사용 가능합니다.

```python
#https://huggingface.co/docs/transformers/model_doc/electra

#single-label classification
model = ElectraForSequenceClassification.from_pretrained("bhadresh-savani/electra-base-emotion", num_labels=num_labels)

#multi-label classification
model = ElectraForSequenceClassification.from_pretrained(
    "bhadresh-savani/electra-base-emotion", num_labels=num_labels, problem_type="multi_label_classification"
)
```

본 프로젝트는 기존 pre-trained 모델을 사용하여 Classification, Summarization등 원하는 Task의 Fine-Tuning 레이어 적용이 가능한지의 여부를 공부하기 위한 테스트 프로젝트입니다.

Fine-Tuning 레이어는 다음과 같은 구조로 적용되었으며 Electra 모델의 top layer를 input으로 전달받아 사용합니다.
<p float="left" align="center">
    <img width="900" src="https://user-images.githubusercontent.com/이미지-추가하기" />  
</p>

```python
#ELECTRA
def forward(self, token_idx, attention_mask):
    hidden_staes =  self.model(input_ids=token_idx, attention_mask=attention_mask.float().to(token_idx.device),
                                  return_dict=False)
    output = hidden_staes[:, 0, :]

    return output
```
```python
#Classication Layer(RNN)
def forward(self, input):
    output, _ = self.rnn(input)
    out_fow = output[range(len(output)),  :self.hidden_size]
    out_rev = output[:, self.hidden_size:]
    output = torch.cat((out_fow, out_rev), 1)
    output = self.dropout(output)

    out_cls = self.active_function(self.wo(torch.squeeze(output, 1)))

    return out_cls
```

### 사용법
config.json : 파일 내에서 다음과 같은 설정이 가능합니다.
- dir_name : 프로젝트 디렉토리명
- chk_point : 체크포인트 파일명
- train_file : 학습용 데이터셋 파일명
- test_file : 테스트 데이터셋 파일명
- tokenizer_len : ELECTRA 토크나이저 토큰길이
- max_length : ELECTRA 토크나이저 토큰길이
- encoder : Fine Tuning Layer 설정(rnn, linear)
- mode : 학습(train) 또는 검증/사용(vali) task 지정
- input_size : Fine Tuning Layer 입력
- hidden_size : Fine Tuning Layer hidden size
- num_layer : Fine Tuning Layer 수(Bidirection)
- num_class : Classification 할 Class 수
- drop_rate_encoder : Fine Tuning Layer drop rate 설정
- drop_rate_electra : ELECTRA drop rate 설정
- batch_size : Batch Size 설정
- warmup_rate : Optimizer Warmup Rate 설정
- epoch : Epoch 설정
- grad_norm : Grad_normalization 설정
- learn_rate : Learning Rate 지정

### Environment
- Colab Python 3.10
- GPU: V100 

### 결과

결과 비교를 위한 Test Dataset은 [Naver sentiment movie corpus](https://github.com/e9t/nsmc) 를 활용하였습니다.


- 작성중
테이블(F1 Score 비교하기)
## Reference
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [Binary Classification](https://github.com/na2na8/ELECTRABinaryClassification)
- [Huggingface Documents](https://huggingface.co/docs/transformers/model_doc/electra)
- [Naver sentiment movie corpus](https://github.com/e9t/nsmc)
