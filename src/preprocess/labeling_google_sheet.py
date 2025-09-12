import os
import json

video_folder_path = "/home/piawsa6000/nas192/datasets/projects/gangnam_innovation/violence_fintuning/TADO_Violence_GangNAM/data/raw/elevator_falldown"

category_data = {
    "elevator_violence_001": "violence", "elevator_violence_002": "violence",
    "elevator_violence_003": "violence", "elevator_violence_004": "violence",
    "elevator_violence_005": "violence", "elevator_violence_006": "violence",
    "elevator_violence_007": "violence", "elevator_violence_008": "violence",
    "elevator_violence_009": "violence", "elevator_violence_010": "violence",
    "elevator_violence_011": "violence", "elevator_falldown_001": "falldown",
    "elevator_falldown_002": "falldown", "elevator_falldown_003": "falldown",
    "elevator_falldown_004": "falldown", "elevator_falldown_005": "falldown",
    "elevator_falldown_006": "falldown", "elevator_falldown_007": "falldown",
    "elevator_falldown_008": "falldown", "elevator_falldown_009": "falldown",
    "elevator_falldown_010": "falldown", "elevator_falldown_011": "falldown",
    "elevator_falldown_012": "falldown", "elevator_falldown_013": "falldown",
    "elevator_falldown_014": "falldown", "elevator_falldown_015": "falldown",
    "elevator_falldown_016": "falldown", "elevator_falldown_017": "falldown",
    "elevator_falldown_018": "falldown", "elevator_falldown_019": "falldown",
    "elevator_falldown_020": "falldown", "elevator_normal_001": "normal",
    "elevator_normal_002": "normal", "elevator_normal_003": "normal",
    "elevator_normal_004": "normal", "elevator_normal_005": "normal",
    "elevator_normal_006": "normal", "elevator_normal_007": "normal",
    "elevator_normal_012": "normal", "elevator_normal_008": "normal",
    "elevator_normal_009": "normal", "elevator_normal_010": "normal"
}

translations = {
    "elevator_violence_001": "As a woman in her 20s is on an elevator, the doors open. A man in his 30s gets on and punches her in the face, causing her to fall down. The man then continuously assaults the fallen woman with his fists and feet.",
    "elevator_violence_002": "A woman in her 20s and a man in his 30s are on an elevator. The man shoves the woman's shoulder from behind, then punches her in the face three times. Although the woman falls and tries to defend herself, the man continues to assault her repeatedly with his fists and feet.",
    "elevator_violence_003": "A woman in her 20s and a man in his 30s are on an elevator. The man shoves the woman's shoulder from behind, then punches her in the face three times. Although the woman falls and tries to defend and resist, the man continues to assault her repeatedly with his fists and feet.",
    "elevator_violence_004": "A man in his 30s is looking at his phone while getting on an elevator. A woman in her 20s appears and punches him in the face. The man falls and tries to defend himself, but the woman continuously hits him with her fists and kicks him.",
    "elevator_violence_005": "A woman in her 20s and a man in his 30s are on an elevator. The man is pressing the buttons while the woman stands behind him. The woman shoves his back shoulder threateningly, and the moment he turns around, she punches him in the face. The man falls into a defensive posture, and the woman corners him, assaulting him with kicks and punches. The man gets up and attempts to resist as much as possible.",
    "elevator_violence_006": "A woman in her 20s and a man in his 30s get on an elevator and talk face-to-face. The man becomes agitated, grabs the woman by her collar threateningly, causing her to fall. He keeps hold of her collar and drags her out when the elevator doors open.",
    "elevator_violence_007": "A woman in her 20s wearing black enters an elevator, followed by a man in his 30s wearing a shirt who grabs her shoulder and starts an argument. Afterwards, the man and woman threaten each other by shoving shoulders and heads. They engage in a mutual assault, grabbing each other's hair before exiting the elevator.",
    "elevator_violence_008": "A woman in her 20s wearing black enters an elevator, followed by a man in his 30s wearing a shirt. After the doors close, they argue, and the man grabs the woman's head and shoves her. The woman kicks the man's leg in retaliation. They slap each other's faces. The man then kicks the woman, causing her to fall down.",
    "elevator_violence_009": "A woman in black is on her phone while boarding an elevator. An unidentified person wearing a bucket hat enters, pulls out a weapon, and threatens her. The startled woman puts her phone in her pocket and attempts to take the weapon. The person kicks at her to keep her away, and she falls to a sitting position. To escape, the woman presses a button and flees.",
    "elevator_violence_010": "A woman and a man, both in black, are on an elevator with the woman standing behind the man. She taps his shoulder, and as soon as he turns, she pulls out a weapon and threatens him. The man is startled and confronts her. The woman swings the weapon, attacking him. After attacking him several times, she flees when the doors open, and the man collapses.",
    "elevator_violence_011": "A woman in purple, a man in white, and a man in grey are on an elevator. The woman grabs the head of the man in white, while the man in grey watches in shock. The woman corners the man in white and kicks him, causing him to sit down on the floor.",
    "elevator_falldown_001": "A permed man in a white shirt is on an elevator. He struggles to support himself and tries to grab the handrail, but fails. He falls, collapsing into a sitting position beneath the elevator buttons.",
    "elevator_falldown_002": "A permed man in a white shirt is on an elevator. As he is about to fall, he tries to brace himself by grabbing the handrail. He groans in pain and collapses forward, toward the elevator doors.",
    "elevator_falldown_003": "A permed man in a white shirt is on an elevator. He clutches his head in pain, loses his balance, and crouches down in the back corner of the elevator.",
    "elevator_falldown_004": "A woman in a white shirt, in apparent distress, loses her balance in the center of the elevator and collapses while holding onto the handrail.",
    "elevator_falldown_005": "A woman in a white shirt stumbles at the front of the elevator, loses her balance, and falls backward. She is wearing black slippers.",
    "elevator_falldown_006": "A woman in a white shirt is at the back of the elevator. She bends over and crouches down. While crouched, she falls forward toward the elevator doors.",
    "elevator_falldown_007": "A man wearing a black tank top and a white t-shirt is at the back of the elevator. He loses his balance, staggers, loses consciousness, and collapses beneath the elevator buttons.",
    "elevator_falldown_008": "A man in his 30s wearing a black tank top and a white t-shirt staggers in the middle of the elevator. He tries to support himself with the back handrail but loses consciousness and collapses.",
    "elevator_falldown_009": "A man in his 30s wearing a black tank top and a white t-shirt, seemingly drunk, stumbles left and right, unable to control his body. He then loses consciousness and collapses beneath the elevator buttons.",
    "elevator_falldown_010": "A woman in black is in the center of the elevator. Unable to keep her balance, she tries to support herself by placing a hand on the door, but collapses in front of the buttons and begins to convulse.",
    "elevator_falldown_011": "A woman in black gets on the elevator clutching her chest. She stumbles, loses consciousness, falls toward the middle-right side of the elevator, and begins to convulse.",
    "elevator_falldown_012": "A woman in black is standing at the front of the elevator and sits down abruptly in front of the doors. Her body sways left and right, then she collapses to her left. Her trembling black sandals suggest she is having a seizure.",
    "elevator_falldown_013": "A man in his 30s wearing a white t-shirt with 'CREW' written on it is bent over, holding the handrail. He clutches his chest in pain and collapses toward the elevator doors.",
    "elevator_falldown_014": "A man in a white t-shirt and black shorts is standing by the right handrail. He clutches his chest in pain and collapses toward the back-right corner of the elevator.",
    "elevator_falldown_015": "A man in a white t-shirt and black shorts is leaning against the elevator wall next to the right handrail. He clutches his chest in pain and collapses toward the back-right corner.",
    "elevator_falldown_016": "A woman wearing a black 'Connecting Roads' shirt and a man in a black tank top and white t-shirt are on an elevator. The woman, in pain, collapses while holding the handrail. The man makes a sound to check her condition, then helps her up by putting her arm around his neck and moves her out of the elevator.",
    "elevator_falldown_017": "A woman wearing a black 'Connecting Roads' shirt and a man in a black tank top and white t-shirt are on an elevator. The man loses consciousness and crouches down. The woman asks about his condition, checks on him, and helps him out of the elevator by supporting his arm.",
    "elevator_falldown_018": "A woman in a grey t-shirt gets on an elevator and collapses in the middle, toward the right. The doors open and a man in his 30s wearing a grey shirt and shorts sees her, looking startled. The man then lifts the collapsed woman and carries her out of the elevator.",
    "elevator_falldown_019": "A man in black shorts and black shoes collapses toward the back of the elevator. The doors open and a woman in grey clothes discovers him. She is startled and surprised but approaches to check his condition. While on the phone, she helps him into a sitting position, then helps him stand up and supports him out of the elevator.",
    "elevator_falldown_020": "A woman in grey, a man in grey, and a man in white are on an elevator. The man in white collapses toward the front. The other two are startled and check on his condition. They help the collapsed man up and support him out of the elevator.",
    "elevator_normal_001": "A woman in grey and another person in white are on an elevator. The person in white (a man) pushes the woman's shoulder. The woman in grey looks startled and annoyed. The man then puts his arm around her shoulder as they exit when the elevator opens.",
    "elevator_normal_002": "A man in white and a woman in grey are on an elevator. The woman touches the man's shoulder; he resists by swinging his arm in mock annoyance. She puts her arm around his shoulder, and he puts his arm around hers as they exit the elevator.",
    "elevator_normal_003": "A woman in grey is looking at her phone while on the elevator. She drops the phone, bends down to pick it up, checks its condition, and then continues to use it.",
    "elevator_normal_004": "A man in a white t-shirt with 'CREW' on it is playing a game on his phone. He loses his grip and drops it on the floor, then bends down to pick it up. He fails to get a good grip and drops it again, bending down to pick it up a second time. He continues playing his game.",
    "elevator_normal_005": "A woman in grey puts her arm around a man in white's shoulders for a hug. The man turns toward her, and they both open their arms to embrace each other. They pat each other's backs and sway from side to side. The man strokes or pulls her hair playfully. Both have happy expressions.",
    "elevator_normal_006": "A woman in grey is exercising, putting her arms behind her head and lifting her legs. She then waves her arms up and down while bending her knees. The elevator doors open to reveal a man in his 30s in a white shirt. The woman looks surprised, turns toward the doors, and starts using her phone. The man gets on and presses a button.",
    "elevator_normal_007": "A woman in grey and a man in white are standing facing each other in an elevator. The woman holds out her palm for a high-five. They repeatedly give each other high-fives. The man tries to hug the woman, and she raises her arms in a victory pose. The doors open and they exit.",
    "elevator_normal_008": "The elevator doors open and a man and a woman, both in light-colored clothes, enter carrying a white desk. After placing it in the center, the man stands behind the woman to the left. The woman stretches her arms. The man and woman both stretch their arms. When the doors open, they exit carrying the desk.",
    "elevator_normal_009": "The elevator opens and a woman in grey enters from the left, while a man in white enters from the right. They playfully high-five. The doors open and the man exits. The woman, now alone, plays a game on her phone and touches her hair.",
    "elevator_normal_010": "A woman in grey and a man in white are on an elevator. The woman touches the man's hair affectionately. The man strokes her hair, and she touches his face. The doors open and the woman exits. The man plays a game on his phone and then neatens his hair. When the doors open again, he also exits.",
    "elevator_normal_012": "The elevator doors open and a man and a woman, both in light-colored clothes, enter carrying a white desk. After placing it in the center, they stand facing each other. The woman swings her hand near the man's ear to catch a bug. They both wave their hands to shoo the bug away."
}
# --------------------------------------------------------------------------


def create_json_labels(target_directory):
    """지정된 디렉토리의 비디오 파일과 매칭되는 JSON 라벨을 생성합니다."""
    if not os.path.isdir(target_directory):
        print(f"오류: '{target_directory}' 경로를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    created_count = 0
    # 지정된 폴더의 모든 파일을 확인
    for filename in os.listdir(target_directory):
        # 파일 이름에서 확장자를 제외한 부분 (예: 'elevator_violence_001')
        base_name, ext = os.path.splitext(filename)

        # 준비된 데이터에 파일 이름이 있는지 확인
        if base_name in category_data:
            # JSON 파일에 들어갈 내용 구성
            json_content = {
                "category": category_data[base_name],
                "description": translations.get(base_name, "No description available.")
            }
            
            # 저장할 JSON 파일의 전체 경로 생성
            json_filepath = os.path.join(target_directory, f"{base_name}.json")
            
            # JSON 파일 저장
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, ensure_ascii=False, indent=4)
            
            print(f"✅ 생성 완료: {base_name}.json")
            created_count += 1
            
    if created_count == 0:
        print("지정된 폴더에 라벨을 생성할 수 있는 비디오 파일을 찾지 못했습니다.")
    else:
        print(f"\n✨ 총 {created_count}개의 JSON 라벨 생성을 완료했습니다.")

# --- 스크립트를 직접 실행할 때 사용되는 부분 ---
if __name__ == '__main__':
    create_json_labels(video_folder_path)