import json
import os

def convert_medical_data(input_file, output_file):
    """
    Medical 도메인 데이터를 다른 도메인과 동일한 형식으로 변환
    
    입력: [["Question: ...", "Answer: ...", "정답"]]
    출력: [{"id": "...", "question": "...", "train_answer": "...", "metric_answer": "..."}]
    """
    print(f"변환 중: {input_file}")
    
    # 기존 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    for i, item in enumerate(data):
        if len(item) == 3:
            # 기존 형식: [question, answer, metric_answer]
            question = item[0]
            train_answer = item[1]
            metric_answer = item[2]
            
            # 새로운 형식으로 변환
            converted_item = {
                "id": f"medical_{i+1:04d}",  # medical_0001, medical_0002, ...
                "question": question,
                "train_answer": train_answer,
                "metric_answer": metric_answer
            }
            converted_data.append(converted_item)
        else:
            print(f"경고: 인덱스 {i}의 항목이 예상된 3개 요소를 가지지 않습니다: {len(item)}개 요소")
    
    # 변환된 데이터를 새 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"변환 완료: {output_file}")
    print(f"총 {len(converted_data)}개 항목 변환됨")
    return len(converted_data)

def main():
    """메인 실행 함수"""
    base_path = "/data/disk5/internship_disk/gyubin/Qwen_MoE/data/processed/medical"
    
    # 변환할 파일들
    files_to_convert = [
        ("medical_train.json", "medical_train_converted.json"),
        ("medical_test.json", "medical_test_converted.json")
    ]
    
    print("=== Medical 도메인 데이터 형식 변환 시작 ===")
    
    total_converted = 0
    
    for input_file, output_file in files_to_convert:
        input_path = os.path.join(base_path, input_file)
        output_path = os.path.join(base_path, output_file)
        
        if os.path.exists(input_path):
            try:
                count = convert_medical_data(input_path, output_path)
                total_converted += count
            except Exception as e:
                print(f"오류 발생 ({input_file}): {e}")
        else:
            print(f"파일을 찾을 수 없습니다: {input_path}")
    
    print(f"\n=== 변환 완료 ===")
    print(f"총 {total_converted}개 항목이 변환되었습니다.")
    
    # 원본 파일을 백업하고 변환된 파일로 교체
    print("\n=== 원본 파일 백업 및 교체 ===")
    for input_file, output_file in files_to_convert:
        input_path = os.path.join(base_path, input_file)
        output_path = os.path.join(base_path, output_file)
        backup_path = os.path.join(base_path, f"{input_file}.backup")
        
        if os.path.exists(output_path):
            # 원본 파일 백업
            if os.path.exists(input_path):
                os.rename(input_path, backup_path)
                print(f"원본 백업: {input_file} -> {input_file}.backup")
            
            # 변환된 파일을 원본 이름으로 변경
            os.rename(output_path, input_path)
            print(f"파일 교체: {output_file} -> {input_file}")
    
    print("\n=== 모든 작업 완료 ===")
    print("원본 파일들은 .backup 확장자로 백업되었습니다.")
    print("변환된 파일들이 원본 이름으로 저장되었습니다.")

if __name__ == "__main__":
    main()
