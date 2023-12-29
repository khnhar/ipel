# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김한나
- 네비게이터 : -
- 리뷰어 : 강다은



# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
      
      <br/>
      
      > 데이터셋 로드, augmentation, 사전학습 모델 로드, 실험 수행, 시각화 등 프로젝트를 수행하는 과정에 필요한 함수를 모두 작성하였습니다.
      > 그러나 데이터셋(input)과 처리 함수간의 크기 및 형식이 맞지 않아 정상적인 작동 및 실험이 불가능하였습니다.
      > 기존 예시 데이터셋이 아닌 새로운 데이터셋을 사용하여, 기존의 예시 함수들과 호환시키는 데 많은 어려움이 있었을 것이라고 생각합니다.
      > 충분한 시간이 주어졌다면 동물이 아닌 식물에 대한 새로운 성능 확인이 가능하였을 것이라고 생각합니다.
      
      <br/>
      
    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
      <br/>
      
      > 다양한 수치 변수가 혼재하여 이해하기 어려운 CutMix 및 MixUp augmentation 함수를 구체적인 주석을 통해 작동 흐름을 이해하기 쉽게 작성하였다.
      
      <br/>
      
      ```
      def mix_2_images(image_a, image_b, x_min, y_min, x_max, y_max):
        # 이미지 텐서를 float32 타입으로 변환
        image_a = tf.cast(image_a, tf.float32)
        image_b = tf.cast(image_b, tf.float32)
    
        # 마스크 텐서 생성 및 float32 타입으로 변환
        mask = tf.zeros_like(image_a, dtype=tf.float32)
    
        # 마스크 업데이트를 위한 영역 지정
        mask_height = y_max - y_min
        mask_width = x_max - x_min
    
        # 업데이트할 영역의 인덱스 생성
        mask_indices = tf.reshape(tf.stack([tf.range(y_min, y_max), tf.range(x_min, x_max)], axis=1), [-1, 2])
        mask_indices = tf.concat([mask_indices, tf.zeros((tf.shape(mask_indices)[0], 1), dtype=tf.int32)], axis=1)
    
        # 마스크 업데이트
        updates = tf.ones((mask_height, mask_width, 3), dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, mask_indices, updates)
    
        # 이미지 혼합
        mixed_image = image_a * (1 - mask) + image_b * mask
        return mixed_image
      ```
      
      <br/>
      
        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
      
      <br/>

      > 기존의 예시 동물 데이터셋이 아닌 새로운 식물 데이터셋을 사용하여 분류 실험을 시도하였다.
      
      <br/>

      <img width="705" alt="image" src="https://github.com/DiANA-KANG/aiffel_quest_khn/assets/149550222/3bde682b-6e39-47ad-9146-456b3afc3cc5">
      
      <br/>

     
- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
      
      <br/>

      > 회고 없음
      
      <br/> 

        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
      <br/>
      
      > 데이터셋 구조를 확인할 수 있는 코드를 함수화하여, train 데이터와 test 데이터에 대하여 확인을 반복하는 코드를 간결하게 구조화하였다.
      
      <br/>
      
      ```
      # 첫 번째 요소를 추출하는 함수
      def check_dataset_structure(ds):
          for example in ds.take(1):
            print("Data type:", type(example))
            if isinstance(example, tuple) and len(example) == 2:
                print("Dataset element is a tuple with two elements (image, label).")
            else:
                print("Dataset element is not a tuple with two elements.")

      # ds_train과 ds_test의 구조 확인
      print("Checking ds_train structure:")
      check_dataset_structure(ds_train)

      print("\nChecking ds_test structure:")
      check_dataset_structure(ds_test)
      ```

      <br/>
      <br/>
     


# 참고 링크 및 코드 개선

