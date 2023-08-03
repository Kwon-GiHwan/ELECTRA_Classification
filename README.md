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

본 프로젝트는 기존 pre-trained 모델을 사용하여 Classification, Summarization등 원하는 Task의 Fine-Tuning 레이어 적용이 가능한지의 여부를 공부하기 위한 테스트입니다.

Fine-Tuning 레이어는 다음과 같은 구조로 적용되었으며 Electra 모델의 top layer를 input으로 전달받아 사용합니다.
<p float="left" align="center">
    <img width="900" src="https://user-images.githubusercontent.com/이미지-추가하기" />  
</p>


## 결과

결과 비교를 위한 Test Dataset은 [Naver sentiment movie corpus](https://github.com/e9t/nsmc) 를 활용하였습니다.

- 작성중

## Reference
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [Binary Classification](https://github.com/na2na8/ELECTRABinaryClassification)
- [Huggingface Documents](https://huggingface.co/docs/transformers/model_doc/electra)
- [Naver sentiment movie corpus](https://github.com/e9t/nsmc)
