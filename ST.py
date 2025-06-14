import streamlit as st
def aiot_system():
    import pytesseract
    import cv2
    import numpy as np
    import whisper
    from openai import OpenAI
    from PIL import Image
    import tempfile
    import streamlit as st
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="api.env")  # ← 指定你的 env 檔名稱
    openai_api_key = os.getenv("OPENAI_API_KEY")
    

    # Whisper 語音辨識模型載入
    whisper_model = whisper.load_model("base")

    # OpenAI API Key 設定
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Streamlit UI
    st.markdown("<h2 style='text-align: left;'>AIoT 智慧互動系統</h2>", unsafe_allow_html=True)
    st.markdown("支援圖片 + 語音 + 文字輸入，多模態 AI 智慧應用")

    # --- 模態選擇 ---
    mode = st.radio("請選擇輸入模式：", ["文字輸入", "影像OCR", "語音輸入"])

    # --- 文字輸入模式 ---
    if mode == "文字輸入":
        user_input = st.text_input("請輸入問題或指令：")
        if st.button("送出") and user_input:
            with st.spinner("GPT 回應中..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": user_input}]
                )
                st.success(response.choices[0].message.content)

    # --- 影像 OCR 模式 ---
    elif mode == "影像OCR":
        uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="上傳的圖片", use_container_width=True)
            text = pytesseract.image_to_string(image, lang='eng+chi_tra')
            st.text_area("OCR 辨識結果：", text)

            if st.button("詢問 GPT 這段文字的意義") and text.strip():
                with st.spinner("GPT 回應中..."):
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"以下內容是 OCR 辨識的文字，請幫我解釋：{text}"}]
                    )
                    st.success(response.choices[0].message.content)

    # --- 語音輸入模式 ---
    elif mode == "語音輸入":
        audio_file = st.file_uploader("請上傳語音檔 (wav/mp3)", type=["wav", "mp3"])
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            st.audio(audio_file)
            with st.spinner("Whisper 正在辨識語音..."):
                result = whisper_model.transcribe(tmp_path)
                st.text_area("語音轉文字：", result["text"])

                if st.button("詢問 GPT") and result["text"].strip():
                    with st.spinner("GPT 回應中..."):
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": result["text"]}]
                        )
                        st.success(response.choices[0].message.content)

    # --- 頁尾標示 ---
    st.markdown("---")
    st.markdown("註：OpenAI的API Key有限制使用額度，一個帳號只有免費提供五塊美金，因此互動過程中如出現Error純屬正常狀況，本組並未額費付費購買額度。")



st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")

# 在側邊欄建立目錄
menu = st.sidebar.selectbox(
    "目錄",
    ["首頁","摘要", "第一章 AIoT 系統整合設計概論", "第二章 物件辨識、LLM 及 ROS2 技術", "第三章 結果與討論", "第四章 結論與心得", "參考文獻", "分工表"]
)

# 根據選項顯示內容
if menu == "首頁":
    # 顯示開頭
    st.markdown("<h1 style='text-align: center;'>113(下)學年度『智慧物聯網系統設計』課程期末報告</h1>", unsafe_allow_html=True)
    st.markdown("---")  # 分隔線

    st.markdown("<h2 style='text-align: center;'>AIoT 系統整合設計與應用-以物件辨識+LLM+ROS2 技術之整合問題為例</h2>", unsafe_allow_html=True)
 
    # 顯示一張圖片(image)
    image_path = "png/0.jpg"
    # 三欄排版，圖片放中間欄
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image_path, use_container_width=True)

    # 顯示其他資訊
    st.markdown("<h2 style='text-align: center;'>指導老師：周榮源</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>班級：碩設計一甲</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>組別：第一組</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>組員：11373106/陳彥碩、11373137/廖翊翔、41023208/廖子儀、41023234/黃仕鈞</h2>", unsafe_allow_html=True)
    st.markdown("---")  # 分隔線
    st.markdown("<h3 style='text-align: center;'>中華民國 114 年 06 月 16 日</h3>", unsafe_allow_html=True)

elif menu == "摘要":
    st.title("摘要")
    st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本專案聚焦於 AIoT 技術在智慧感知、自主決策與邊緣運算等面向的應用實作，依據三大主題進行整合性開發，涵蓋物件偵測、
                語意理解與機構控制等領域，展現跨技術整合能力與創新潛力。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                三大主題分別從視覺感知、多模態互動與機械自動化出發，結合邊緣 AI、感測器與雲端可視化平台，提出具創新性與實作可行性的 AIoT 應用，
                為未來智慧工廠與人機協作奠定基礎。
                </p>
                <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                主題一：YOLO + SAM 演算法應用實作
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本題以 Raspberry Pi 為邊緣平台，實作 YOLO（You Only Look Once）與 SAM（Segment Anything Model）模型，
                針對指定影片進行物體計數、分割與尺寸估計。藉由前處理優化與模型調整，提升辨識準確率與執行效率。最終辨識結果透過 MQTT 
                傳送至 Streamlit 前端，即時展示目標類別與分析資訊。
                </p>
                <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                主題二：LLM 整合與應用實作
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本題以 Streamlit 建構一個支援文字、影像與語音的多模態使用者介面，整合大型語言模型（LLM）、語音辨識（ASR）、
                文字辨識（OCR）與圖像理解功能，構建可互動的 AI 輔助系統。
                </p>
                <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                主題三：AIoT 系統設計與應用（五軸 3D 列印手臂自動換噴頭系統）
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本題設計一套結合 AIoT 與機構自動化的創新應用：針對五軸 3D 列印機中的機械手臂裝設視訊鏡頭，利用 YOLO 進行噴頭套件目標定位
                ，再以 SAM 進行形狀與邊界分割，精準估算套件在工作區外的位置與姿態。系統可根據辨識結果控制手臂完成自動更換噴頭流程，實現模組化
                、自主化與無人化的列印套件管理，解決多材質列印中繁瑣的人工換件問題，展現 AIoT 與智慧製造的深度整合應用。
                </p>
                """, unsafe_allow_html=True)
    
elif menu == "第一章 AIoT 系統整合設計概論":
    
    submenu = st.sidebar.radio("關於選單", ["1.1 AIoT 技術概論", "1.2 AIoT 應用", "1.3 實作內容簡介"])
    if submenu == "1.1 AIoT 技術概論":
        st.title("1.1 AIoT 技術概論")
        st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                AIoT是人工智慧（AI）與物聯網（IoT）兩大技術的結合，
                旨在讓物聯網裝置具備智慧判斷與自主決策的能力。隨著感測器、通訊模組與雲端運算的普及，傳統的 IoT 僅能進行資料收集
                與傳輸，缺乏即時分析與反應的能力。而 AI 的導入，則賦予系統「思考」與「學習」的能力，實現從「連網」走向「智慧」的進化。
                </p>
                    
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                AIoT 的核心架構通常包括四個層級：感測層（資料蒐集）、網路層（資料傳輸）、運算層（資料分析與決策）、應用層（實際應用場景）。
                舉例來說，一個智慧工廠中的 AIoT 系統可以即時監控機台溫度與震動，利用 AI 模型預測異常狀況，並透過邊緣運算器做出快速反應，
                如停機檢修、調整參數，達到預防性維護與效率最佳化。
                </p>

                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                AIoT 的應用橫跨多個領域，包括智慧製造、智慧城市、智慧醫療、智慧零售與智慧交通等。例如，在智慧農業中，透過感測器蒐集土壤濕度
                、氣候等資訊，再透過 AI 分析作物需求，自動調節灌溉與施肥策略，提升產量與資源使用效率。
                </p>

                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                儘管 AIoT 發展潛力無窮，但仍面臨許多挑戰，包括資料隱私與安全問題、異質裝置的整合性、邊緣運算效能限制等。
                此外，AI 模型的準確性與可解釋性，也攸關系統的可靠度與用戶信任。
                </p>

                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                AIoT 是邁向未來智慧世界的重要關鍵技術。它不僅提升了物聯網系統的自主性與智能化，也促進了數據驅動的決策模式，
                正在悄然重塑我們的生活與產業結構。未來，隨著硬體性能提升與 AI 演算法的演進，AIoT 將更加普及，成為數位轉型不可或缺的基石。
                </p>    
                """, unsafe_allow_html=True)

    elif submenu == "1.2 AIoT 應用":
            st.title("1.2 AIoT 應用")
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                AIoT（Artificial Intelligence of Things）結合了人工智慧（AI）與物聯網（IoT）技術，讓終端裝置不再只是
                「傳送資料的感測器」，而能「即時思考與判斷」，實現更高層次的智慧化應用。隨著晶片小型化與 AI 模型輕量化的發展，
                如 TinyML（Tiny Machine Learning）與 Edge Impulse 等技術平台，AI 模型得以部署在資源受限的邊緣裝置上，
                無須依賴雲端即可完成即時分析與決策，提升反應速度與隱私保護能力。
                </p>
                        
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                TinyML 是一種專為微控制器（如 Arduino、ESP32 等）設計的機器學習技術，透過優化模型大小與運算效率，使裝置能以極低功耗執行推論任務。例如：在一個配備音訊感測器的工業風扇上，使用 TinyML 可即時偵測異常聲音，並在出現
                異常模式時發出警報，達到預警效果。
                </p>
                        
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                Edge Impulse 則是一個支援邊緣 AI 模型開發的雲端平台，提供使用者從資料採集、訓練、驗證到部署的一條龍服務，
                適合無 AI 背景的開發者快速上手。透過 Edge Impulse，工廠工程師可以將震動感測器蒐集的數據快速訓練成模型，
                部署到微控制器中進行即時機械診斷，大幅縮短開發時程。
                </p>
                        
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                以 智慧工廠 為例，某些工業場域中部署了數百個感測節點來監控馬達、壓縮機與輸送帶的狀況。過去必須集中回傳到雲端才能分析
                ，但現在可在邊緣裝置中直接執行 AI 模型，判斷設備是否異常，進行即時反應（例如自動停機或回報異常），同時減少資料傳輸量與延遲
                ，提升生產效率與系統穩定性。
                </p>
                        
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                AIoT 正透過 TinyML、Edge AI 等技術實現「智慧在地化」，讓終端裝置變得更聰明、更即時、更節能。這種轉變不僅提升
                了工業現場的自主管理能力，也為未來萬物智慧鋪路，成為數位轉型的重要推手。
                </p>
                """, unsafe_allow_html=True)

    elif submenu == "1.3 實作內容簡介":
            st.title("1.3 實作內容簡介")
            st.markdown("""
                <p style="text-indent: 0em; line-height: 1.8; font-size: 25px; text-align: justify;">
                主題一：辨識鍵盤上按鍵的數量，以Yolo訓練模型，並使用SAM2進行按鍵的影像分割，最後計算分割數量從而得出按鍵數量。
                </p>
                
                <p style="text-indent: 0em; line-height: 1.8; font-size: 25px; text-align: justify;">
                主題二：利用Streamlit寫一個網頁，內含AIoT 智慧互動系統，提供即時的文字輸入互動、影像OCR和語音輸入等功能。
                </p>
                        
                <p style="text-indent: 0em; line-height: 1.8; font-size: 25px; text-align: justify;">
                主題三：將五軸3D列印中的機械手臂裝上視訊鏡頭，並利用YOLO 與 SAM 功能使其能辨識工作區外噴頭套件之位置與距離，並自行更換套件。
                </p>
                """, unsafe_allow_html=True)

elif menu == "第二章 物件辨識、LLM 及 ROS2 技術":
    
    submenu = st.sidebar.radio("關於選單", ["2.1 物件辨識技術技術", "2.2 LLM", "2.3 ROS2", "2.4 設計問題一", "2.5 設計問題二", "2.6 設計問題三"])
    if submenu == "2.1 物件辨識技術技術":
        st.title('2.1 物件辨識技術技術')
        st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                物件辨識是一種電腦視覺技術，目的是讓機器能從影像或影片中「看懂」並辨認特定物體的種類與位置。
                其核心在於將圖像轉換為數值特徵，並結合機器學習或深度學習方法進行分類與定位。
                </p>
                <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                原理與方法
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                物件辨識是一種電腦視覺技術，目的是讓機器能從影像或影片中「看懂」並辨認特定物體的種類與位置。
                其核心在於將圖像轉換為數值特徵，並結合機器學習或深度學習方法進行分類與定位。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                隨著深度學習的發展，卷積神經網路成為物件辨識的主流技術。CNN 透過多層神經網路自動學習圖像中的階層特徵，
                使系統具備更強的泛化能力與準確率。其中，YOLO系列演算法以其「即時辨識」與「單階段輸出」
                的特性廣受應用；另一種代表是 Faster R-CNN，雖準確率高但運算速度較慢。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                針對分割需求，也可結合如 SAM 這類語義分割工具，進一步將物件輪廓與背景區隔出來，提升系統對環境的理解能力。
                </p>
                <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                應用範疇
                </p>
                <p style="text-indent: 0em; line-height: 1.8; font-size: 25px; text-align: justify;">
                物件辨識廣泛應用於多個領域：
                </p>

                <ul style="font-size: 23px; line-height: 1.8; text-align: justify;">

                <li>
                <b>智慧製造：</b>在工廠中用於自動瑕疵檢測、零件辨識與品質控制。
                </li>

                <li>
                <b>智慧交通：</b>辨識車牌、行人與車輛類型，應用於自駕車與交通監控。
                </li>

                <li>
                <b>醫療影像：</b>分析 X 光、MRI 影像，自動標記異常組織或病變部位。
                </li>

                <li>
                <b>零售業：</b>自助結帳機器能即時辨識商品，提升顧客體驗。
                </li>

                <li>
                <b>農業：</b>辨識果實成熟度、害蟲種類與作物種類，實現精準農業。
                </li>

                </ul>
                
        """, unsafe_allow_html=True)


    elif submenu == "2.2 LLM":
            st.title('2.2 LLM')
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                大型語言模型是人工智慧領域中以自然語言處理為核心的深度學習模型，能理解並產生與人類語言相近的文字內容。
                其基本原理是透過大量語料訓練出數百億至數兆參數的神經網路，使模型能根據上下文預測下個詞語或完成語意任務。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                傳統 NLP 方法如規則式系統或基於機器學習的詞袋模型，在語意理解上表現有限。而 LLM 採用的關鍵技術是「Transformer」架構，
                透過自注意力機制捕捉語句中各詞之間的長距離依賴關係。代表性模型包括 GPT（Generative Pre-trained Transformer）
                、BERT（Bidirectional Encoder Representations from Transformers）等。GPT 系列屬於生成式模型，
                能進行摘要、翻譯、對話生成等任務；而 BERT 擅長分類與問答任務。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                LLM 的訓練流程可分為三階段：首先是預訓練，使用大量未標註語料學習語言結構與知識；接著為微調，針對特定任務進行監督式學習
                ；最後則可加入人類回饋訓練（如 RLHF）提升回應品質與安全性。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                應用層面，LLM 已廣泛滲透至日常生活與產業實務中。例如，客服機器人可即時理解並回應使用者問題；文件摘要與自動翻譯提升知識工作效率
                ；醫療領域應用於電子病歷摘要與診斷輔助；教育領域則可打造智慧助教，根據學生提問進行即時解答。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                更進一步，LLM 還可整合多模態輸入（如文字＋圖像＋語音），成為所謂 VLA（Vision-Language-Action）模型，開啟更多跨領域 AIoT 應用。
                搭配語音辨識、圖像辨識、感測器資料等，LLM 可作為多模態中樞，實現真正的智慧助理與語意決策中心。
                </p>
                """, unsafe_allow_html=True)                        


    elif submenu == "2.3 ROS2":
            st.title('2.3 ROS2')
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                ROS 2是針對現代機器人應用所設計的開源軟體框架，為 ROS 1 的下一代版本。雖名為「作業系統」，但實際上 ROS 是一組模組化軟體工具與通訊中介，
                協助開發者更有效率地開發與管理複雜的機器人系統。其核心精神在於模組解耦、分散式架構與通訊中立性，讓軟體開發能專注於功能邏輯，
                不必重複處理低階通訊與資源管理。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                ROS 2 的設計核心建立在 DDS 通訊協定之上，提供即時、可靠、安全的分散式通訊能力，支援多平台與跨網段資料交換，適合應用於工業與實務場域。
                與 ROS 1 相比，ROS 2 更加注重系統穩定性、效能擴充性與安全性，並支援多執行緒、多語言開發（如 C++、Python）與實時作業系統（如 RTOS）。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                ROS 2 的開發方法以「節點」為基本單位，所有感測器讀取、資料處理、決策邏輯與控制輸出皆封裝於各自的節點中
                ，彼此透過話題、服務、動作進行非同步通訊。開發者可利用現有的功能套件快速建構感知、地圖建構、導航與操作模組
                ，並透過 launch file 整合整體系統流程。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                應用面上，ROS 2 已廣泛運用於自走車、協作機器人、無人機、醫療機器人與工業自動化系統。例如，在智慧工廠中可用於協調多機械手臂的動作流程
                ；在物流領域中則可導入於自動搬運車以實現自主導航與動態避障。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                ROS 2 是建構機器人智慧系統的基礎骨幹，不僅加速原型開發，也支援商業落地所需的穩定性與擴充性。隨著 AIoT 與邊緣計算的進展
                ，ROS 2 將成為智慧機器與分散控制系統中的關鍵核心。
                </p>
                """, unsafe_allow_html=True)


    elif submenu == "2.4 設計問題一":
            st.title('2.4 設計問題一：')
            st.markdown("""
                <p style="text-indent: 0em; line-height: 1.8; font-size: 25px; text-align: justify;">
                說明：辨識鍵盤上按鍵的數量，以Yolo訓練模型，並使用SAM2進行按鍵的影像分割，最後計算分割數量從而得出按鍵數量。
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                程式碼
                </p>
                """, unsafe_allow_html=True)    
            code = """
import cv2
from ultralytics import YOLO
import numpy as np

yolo_model = YOLO("yolo11n.pt")  # 你的模型路徑

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]

    keyboard_boxes = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 0:  # 你模型中的鍵盤 class
            keyboard_boxes.append(box.cpu().numpy().astype(int))

    for (x1, y1, x2, y2) in keyboard_boxes:
        keyboard_roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(keyboard_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        key_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 2000:
                x, y, w, h = cv2.boundingRect(cnt)

                # 過濾不合理長寬比的輪廓（如手指）例：長條形或扁平條
                aspect_ratio = w / h if h != 0 else 0
                if 0.4 < aspect_ratio < 2.5:
                    cv2.rectangle(keyboard_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    key_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Keys: {key_count}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO + Contour Key Count (filtered)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
        """
            with st.expander("點擊展開完整程式碼"):
                 st.code(code, language="python")

            st.markdown("""
            <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                架構圖
                </p>
                """, unsafe_allow_html=True)
            st.image("png/p1.png")


    elif submenu == "2.5 設計問題二":
            st.title('2.5 設計問題二')
            st.markdown("""
                <p style="text-indent: 0em; line-height: 1.8; font-size: 25px; text-align: justify;">
                說明：利用Streamlit寫一個網頁，內含AIoT 智慧互動系統，提供即時的文字輸入互動、影像OCR和語音輸入等功能。
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                程式碼
                </p>
                """, unsafe_allow_html=True) 
            code = """
    import pytesseract
    import cv2
    import numpy as np
    import whisper
    from openai import OpenAI
    from PIL import Image
    import tempfile
    import streamlit as st
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env")  # ← 指定你的 env 檔名稱
    openai_api_key = os.getenv("OPENAI_API_KEY")
    

    # Whisper 語音辨識模型載入
    whisper_model = whisper.load_model("base")

    # OpenAI API Key 設定
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Streamlit UI
    st.markdown("<h2 style='text-align: left;'>AIoT 智慧互動系統</h2>", unsafe_allow_html=True)
    st.markdown("支援圖片 + 語音 + 文字輸入，多模態 AI 智慧應用")

    # --- 模態選擇 ---
    mode = st.radio("請選擇輸入模式：", ["文字輸入", "影像OCR", "語音輸入"])

    # --- 文字輸入模式 ---
    if mode == "文字輸入":
        user_input = st.text_input("請輸入問題或指令：")
        if st.button("送出") and user_input:
            with st.spinner("GPT 回應中..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": user_input}]
                )
                st.success(response.choices[0].message.content)

    # --- 影像 OCR 模式 ---
    elif mode == "影像OCR":
        uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="上傳的圖片", use_container_width=True)
            text = pytesseract.image_to_string(image, lang='eng+chi_tra')
            st.text_area("OCR 辨識結果：", text)

            if st.button("詢問 GPT 這段文字的意義") and text.strip():
                with st.spinner("GPT 回應中..."):
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"以下內容是 OCR 辨識的文字，請幫我解釋：{text}"}]
                    )
                    st.success(response.choices[0].message.content)

    # --- 語音輸入模式 ---
    elif mode == "語音輸入":
        audio_file = st.file_uploader("請上傳語音檔 (wav/mp3)", type=["wav", "mp3"])
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            st.audio(audio_file)
            with st.spinner("Whisper 正在辨識語音..."):
                result = whisper_model.transcribe(tmp_path)
                st.text_area("語音轉文字：", result["text"])

                if st.button("詢問 GPT") and result["text"].strip():
                    with st.spinner("GPT 回應中..."):
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": result["text"]}]
                        )
                        st.success(response.choices[0].message.content)

    # --- 頁尾標示 ---
    st.markdown("---")
    st.markdown("註：OpenAI的API Key有限制使用額度，一個帳號只有免費提供五塊美金，因此互動過程中如出現Error純屬正常狀況，本組並未額費付費購買額度。")        
            """
            with st.expander("點擊展開完整程式碼"):
                 st.code(code, language="python")

            st.markdown("""
            <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                架構圖
                </p>
                """, unsafe_allow_html=True)
            st.image("png/p2.png")


    elif submenu == "2.6 設計問題三":
            st.title('2.6 設計問題三')
            st.markdown("""
                <p style="text-indent: 0em; line-height: 1.8; font-size: 25px; text-align: justify;">
                說明：將五軸3D列印中的機械手臂裝上視訊鏡頭，並利用YOLO 與 SAM 功能使其能辨識工作區外噴頭套件之位置與距離，並自行更換套件。
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                架構圖
                </p>
                """, unsafe_allow_html=True)
            st.image("png/000.png")

elif menu == "第三章 結果與討論":
    
    submenu = st.sidebar.radio("關於選單", ["3.1 設計問題一", "3.2 設計問題二", "3.3 設計問題三"])
    if submenu == "3.1 設計問題一":

        st.title('成果展示')
        st.image("png/1.png")
        st.image("png/2.png")
        st.video("video/NO_1.mp4") 

        st.title('主題一討論')
        st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本主題雖完成初始設定之目標，即「辨識鍵盤上按鍵的數量，以Yolo訓練模型，並使用SAM2進行按鍵的影像分割，最後計算分割數量從而得出按鍵數量」
                但在過程中SAM2套件不斷出現問題，SAM2套件在安裝後發現其所需要的torch套件版本為2.5.1版，此版本過於新穎，導致Python、PyTorch與CUDA的
                版本都需要更新至各自的特定版本中才可以相容。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                更新過程在不斷除錯，使各軟體可以相容的過程中發現，在SAM2的Github內容的更新日誌中可以發現其要求的某些版本在官網上還沒開始發行，
                因此全部套件再重新降低版本，尋找可以相容的版本組合，在過程中來來回回的安裝與卸載，意外導致了殘存版本檔案間的互相影響，致使問題過於複雜，
                並且最後無法解決問題。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                最後鑒於本題即使使用SAM2模型進行影像分割，最終樹梅派的性能也無法負擔即時影像辨識、分割與計數的需求，因此最終只採用影像辨識+計數的方法求出按鍵數量。
                </p>
                """, unsafe_allow_html=True) 


    elif submenu == "3.2 設計問題二":
            st.title('成果展示')
            st.markdown("---")
            aiot_system()  # 呼叫系統介面

            st.title('主題二討論')
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本題成果如下方展示之AI互動系統，背後的語言模型為ChatGPT，唯一美中不足的是剛開始不知道GPT
                的API Key需要付費，因此在測試期間即將免費的額度使用完畢，若互動過程中如出現Error純屬正常狀況，本組並未額費付費購買額度。
                </p>
                """, unsafe_allow_html=True)


    elif submenu == "3.3 設計問題三":
            st.title('3.3 設計問題三')
            st.title('主題三討論')
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                本題設計一套結合 AIoT 與機構自動化的創新應用：針對五軸 3D 列印機中的機械手臂裝設視訊鏡頭，利用 YOLO 進行噴頭套件目標定位，
                再以 SAM 進行形狀與邊界分割，精準估算套件在工作區外的位置與姿態。系統可根據辨識結果控制手臂完成自動更換噴頭流程，實現模組化
                、自主化與無人化的列印套件管理，解決多材質列印中繁瑣的人工換件問題，但目前尚未設計完成，因此沒有進行實體展示。
                </p>
                """, unsafe_allow_html=True)
            
elif menu == "第四章 結論與心得":
    
    submenu = st.sidebar.radio("關於選單", ["4.1 結論", "4.2 心得"])
    if submenu == "4.1 結論":
        st.title('4.1 結論')
        st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                在這次的期末報告中，大家都感受到需要在智慧物聯網系統中投入更多努力，這學期所學的技術在應用層面上的潛力與挑戰並存。
                在主題一中，團隊嘗試結合 YOLO 模型與 Meta 的 SAM2 分割技術，進行鍵盤按鍵的計數。雖然在初步達成目標，但在實務操作中發現 SAM2 的
                套件相依性極高，且與目前主流的 Python 與 CUDA 版本難以相容，加上版本更新過於頻繁與不穩定，導致開發過程中多次遭遇推論失敗。
                即便技術理論成熟，Raspberry Pi 的算力也難以支撐 SAM2 所需的運算量，因此最終回歸僅以 YOLO 模型進行即時辨識與輪廓分析的策略。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                主題二則為一組以 ChatGPT 為核心的人機互動系統，雖功能完整，惟受到 GPT API 使用門檻與計費限制的影響，測試階段即耗盡免費額度，
                使得展示過程可能遭遇連線錯誤，反映出現階段AI服務商業模式對開發端的限制，也提示開發者未來在系統部署時需考量 API 成本與流量調度策略。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                主題三則將 AIoT 結合於機械手臂的模組化，運用 YOLO 定位噴頭，再以 SAM 分割分析幾何形狀，進一步驅動五軸列印手臂完成自主換件動作。
                該設計展現了 AI 在工業自動化中的實質價值與可擴展性，雖因尚未完成而無實體展示，但初步認為概念具未來發展潛力。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                最後，三個主題均遇到 AI 在實際應用中從模型選型、系統整合到硬體限制的多層次挑戰，未來若能更精確掌握工具的適用性、
                資源配置與開發流程規劃，將能進一步提升 AI 應用的實用性與可落地性。
                </p>
                """, unsafe_allow_html=True)

    elif submenu == "4.2 心得":
            st.title("11373106/陳彥碩/心得")
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                上完這學期的課後，我收穫良多，雖然過程中跟上學期的智慧機械設計期末報告一樣狀況百出，程式似乎總有修不完的bug，但幸好最終大部分
                都有順利解決，不過也是要經過實作才能更好的累積自己的實力，且經過實作後對於整個智慧物聯網系統都有更深入的了解，
                尤其是在學習的過程中我發現很多有趣的智慧物聯網技術或許是可以應用在我的論文題目裡，好比主題三裡提到的關於五軸 3D 列印機中的機械手臂裝設視訊鏡頭
                的AIoT 與機構自動化的創新應用，雖然還沒完全做出來，但我認為它是個需要加在五軸 3D 列印機中的功能，否則要更換不同口徑的進行列印會是一件繁瑣且困難的事情。
                </p>
                """, unsafe_allow_html=True)
            
            st.title("11373137/廖翊翔/心得")
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                這學期透過智慧物聯網系統設計這門課，我學習到許多實用的技術與概念，包括大型語言模型（LLM）的應用、YOLO影像辨識，以及使用樹
                莓派進行實作。這些內容不僅拓展了我對物聯網與人工智慧整合的理解，也讓我對實際應用有更深的感受。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                整體來說，這門課強調實作與整合能力，幫助我建立了跨領域思考的基礎，對我未來從事 AIoT 或智慧製造等領域具有很大的幫助。
                </p>
                """, unsafe_allow_html=True)
            
            st.title("41023208/廖子儀/心得")
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                在實作 YOLO 結合 SAM 的應用中，我體會到即時物件偵測與精確語意分割的強大互補性，能有效提升視覺辨識的完整度與準確性。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                進一步整合 LLM（大型語言模型）後，不僅可以自然語言解釋影像辨識結果，還能結合邏輯推理與使用者互動，拓展應用層面。
                在 AIoT 系統設計中，將感測器、機器人與雲端 AI 結合，讓我深刻理解跨域整合的挑戰與價值。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                透過智慧物聯網系統設計課程，我學會如何將視覺 AI、語言理解與硬體控制整合為一個具體系統，展現人工智慧在現實環境中的應用潛力與
                未來發展可能。
                </p>
                """, unsafe_allow_html=True)
            
            st.title("41023234/黃仕鈞/心得")
            st.markdown("""
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                在這次《智慧物聯網系統設計》課程中，我學習到物聯網系統的整體架構與應用，包括感測器使用、通訊模組整合、微控制器程式設計，
                以及資料雲端儲存與應用。
                </p>
                <p style="text-indent: 2em; line-height: 1.8; font-size: 25px; text-align: justify;">
                在實作過程中，我實際操作了如 Arduino、ESP32 等開發板，學習如何透過程式控制感測器讀取數據，並利用網路模組將資料傳輸至雲端
                平台進行分析與應用。
                </p>
                """, unsafe_allow_html=True)

elif menu == "參考文獻":
    st.title("參考文獻")
    st.markdown('### 1.[streamlit程式庫](https://cheat-sheet.streamlit.app/)')
    st.markdown('### 2.[USB_cam安裝](https://github.com/as985643/usb_cam.git)')
    st.markdown('### 3.[OpenAI API keys](https://ithelp.ithome.com.tw/articles/10333740)')
    st.markdown('### 4.[CUDA安裝方法](https://qqmanlin.medium.com/cuda-%E8%88%87-cudnn-%E5%AE%89%E8%A3%9D-e982d92162af)')
    st.markdown('### 5.[PyTorch安裝](https://pytorch.org/get-started/previous-versions/)')
    st.markdown('### 6.[Segment Anything in High Quality (HQ-SAM)](https://www.youtube.com/watch?v=UGlEU52wGwM&list=PLGTvxhgE_-gZKRbYQlE6HyUQQZ0wnYMm4&index=1)')
    st.markdown('### 7.[Segment Anything Model 2](https://www.youtube.com/watch?v=toFiUqjWCFw&list=PLGTvxhgE_-gZKRbYQlE6HyUQQZ0wnYMm4&index=3)')
    st.markdown('### 8.[Vision Language Models](https://huggingface.co/blog/vlms-2025)')

elif menu == "分工表":
    st.markdown("""
        <style>
            .center {
                text-align: center;
            }
            
            table {
                margin: 0 auto;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: center;
            }
            .signature {
                text-align: right;
            }
            br {
                font-size: 20px;
                }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="center">
            <h3>『智慧物聯網系統設計』</h2>
            <h4>學期團隊作業/專案設計</h3>
        </div>

        <p class="center">
            課號：0252(碩設計一甲) <br>
            學年：113年度第2學期<br>
            組別：第一組<br>
            題目：AIoT 系統整合設計與應用-以物件辨識+LLM+ROS2 技術之整合問題為例<br>
            成員：11373106/陳彥碩、11373137/廖翊翔、41023208/廖子儀、41023234/黃仕鈞<br>
        </p>

        <div class="center">
            <table>
                <tr>
                    <th>項次</th>
                    <th>學號</th>
                    <th>姓名</th>
                    <th>分工內容</th>
                    <th>貢獻度</th>
                </tr>
                <tr>
                    <td>1</td>
                    <td>11373106</td>
                    <td>陳彥碩</td>
                    <td>全篇主要程式架構撰寫、除錯與網頁報告統整</td>
                    <td>25%</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td>11373137</td>
                    <td>廖翊翔</td>
                    <td>YOLO的照片標註、報告討論、成果攝影與拍照</td>
                    <td>25%</td>
                </tr>
                <tr>
                    <td>3</td>
                    <td>41023208</td>
                    <td>廖子儀</td>
                    <td>輔助全篇程式撰寫、報告討論、與報告構思</td>
                    <td>25%</td>
                </tr>
                <tr>
                    <td>3</td>
                    <td>41023234</td>
                    <td>黃仕鈞</td>
                    <td>YOLO的照片標註、步驟截圖與報告構思</td>
                    <td>25%</td>
                </tr>
            </table>
        </div>

        <p class="center">
            貢獻度總計為100%，請自行核算。<br>
            完成日期：<u>114年06月16日</u>
        </p>

        <p class="center">
            <b>說明</b><br>
            本人在此聲明，本設計作業皆由本人與同組成員共同獨立完成，並無其他第三者參與作業之進行，
            若有抄襲或其他違反正常教學之行為，自願接受該次成績以零分計。同時本人亦同意在上述表格中所記載之作業貢獻度，
            並以此計算本次個人作業成績。
        </p>

        <p class="indent">
            成員簽名：
        </p>
        """, unsafe_allow_html=True)
    st.image("png/N1.png")
    st.image("png/N2.png")
    st.image("png/N3.jpg")
    st.image("png/N4.jpg")
