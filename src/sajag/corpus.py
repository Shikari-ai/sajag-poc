"""Synthetic SMS / chat-message corpus: Indian scam families with paired legitimate lookalikes.

Every topic that scammers imitate (KYC, electricity, parcels, challans, reward points, loans,
UPI, jobs, family, OTP) carries BOTH a scam and a genuine version, so the classifier cannot
win by spotting the topic — it has to spot the manipulation. Four languages: English,
romanised Hinglish, Hindi (Devanagari) and Telugu. The Hindi and Telugu templates are
machine-authored and need native-speaker review.

Each template is filled with random slot values several times. Rows generated from the same
template share a ``template_id``; evaluation groups on it so no wording is ever seen in both
train and test.
"""

from __future__ import annotations

import random
import string

# (display name, domain token, official site)
BANKS = [
    ("SBI", "sbi", "https://www.onlinesbi.sbi"),
    ("HDFC Bank", "hdfc", "https://www.hdfcbank.com"),
    ("ICICI Bank", "icici", "https://www.icicibank.com"),
    ("Axis Bank", "axis", "https://www.axisbank.com"),
    ("Kotak", "kotak", "https://www.kotak.com"),
    ("Canara Bank", "canara", "https://canarabank.com"),
    ("PNB", "pnb", "https://www.pnbindia.in"),
]
LOOKALIKE = {
    "sbi": "onlinesbi.co", "hdfc": "hdfcbank.co", "icici": "icicibnak.com",
    "axis": "axisbnk.com", "kotak": "kotakk.com", "canara": "canarabank.co", "pnb": "pnbindia.co",
}
COURIERS = ["India Post", "Blue Dart", "Delhivery", "DTDC", "FedEx"]
SHOPS = ["Amazon", "Flipkart", "Myntra", "Swiggy", "Zomato", "Meesho"]
NAMES = ["Ravi", "Priya", "Anil", "Sneha", "Kiran", "Lakshmi", "Arjun", "Fatima", "Suresh", "Divya"]
AREAS = ["Madhapur", "Kukatpally", "Ameerpet", "Gachibowli", "Secunderabad", "LB Nagar", "Miyapur"]
SCAM_TLDS = ["xyz", "top", "click", "info", "live", "online", "site", "in", "co"]
SCAM_SITES = ["indiapost-redelivery", "echallan-parivahan", "bill-update-portal",
              "reward-redeem-center", "kbc-winner-claim", "loan-instant-approval"]

# topic -> label ("scam" / "legit") -> lang -> templates
TEMPLATES: dict[str, dict[str, dict[str, list[str]]]] = {
    "kyc": {
        "scam": {
            "en": [
                "Dear Customer, your {bank} account will be BLOCKED today due to pending KYC. Update your PAN immediately: {scam_link}",
                "{bank} ALERT: Your netbanking is suspended. Complete e-KYC within 24 hrs to avoid permanent closure. Click {scam_link}",
                "Your {bank} NetBanking access expires tonight. Call KYC officer on {phone} to reactivate. Do not ignore.",
            ],
            "hinglish": [
                "Priya grahak, aapka {bank} khata aaj band ho jayega. KYC update karne ke liye turant is link par click kare {scam_link}",
                "Aapka PAN card bank account se link nahi hai. Account block hone se pehle {phone} par call kare.",
            ],
            "hi": ["प्रिय ग्राहक, आपका {bank} खाता आज बंद कर दिया जाएगा। तुरंत KYC अपडेट करें: {scam_link}"],
            "te": ["ప్రియమైన కస్టమర్, మీ {bank} ఖాతా ఈరోజు బ్లాక్ చేయబడుతుంది. వెంటనే KYC అప్డేట్ చేయండి: {scam_link}"],
        },
        "legit": {
            "en": [
                "Dear Customer, as per RBI guidelines please update your KYC by visiting your nearest {bank} branch or through the official app. {bank} never asks for OTP, PIN or passwords.",
                "Your KYC update request has been received and will be processed in 3 working days. Ref no {ref}. -{bank}",
            ],
            "hinglish": ["Priya grahak, apna KYC update karne ke liye apni nazdeeki {bank} branch par jaye. Bank kabhi OTP ya PIN nahi maangta."],
            "hi": ["प्रिय ग्राहक, अपना KYC अपडेट करने के लिए अपनी नजदीकी {bank} शाखा पर जाएं। बैंक कभी OTP या PIN नहीं मांगता।"],
            "te": ["ప్రియమైన కస్టమర్, మీ KYC అప్డేట్ కోసం దగ్గరలోని {bank} బ్రాంచ్‌ను సందర్శించండి. బ్యాంక్ ఎప్పుడూ OTP లేదా PIN అడగదు."],
        },
    },
    "electricity": {
        "scam": {
            "en": [
                "Dear consumer, your electricity power will be disconnected tonight at {time} because your last month bill was not updated. Please contact our electricity officer {phone} immediately.",
                "ELECTRICITY BILL: Your connection {usc} is due for disconnection today. Pay pending amount now to avoid cut: {scam_link}",
            ],
            "hinglish": ["Priya upbhokta, aaj raat {time} baje aapki bijli kaat di jayegi kyunki pichhle mahine ka bill update nahi hua. Turant bijli adhikari se sampark kare {phone}"],
            "hi": ["प्रिय उपभोक्ता, आज रात {time} बजे आपकी बिजली काट दी जाएगी क्योंकि पिछले महीने का बिल अपडेट नहीं हुआ है। तुरंत बिजली अधिकारी से संपर्क करें {phone}"],
            "te": ["ప్రియమైన వినియోగదారుడా, మీ గత నెల బిల్లు అప్డేట్ కాలేదు కాబట్టి ఈ రాత్రి {time} కి మీ కరెంట్ కట్ చేయబడుతుంది. వెంటనే విద్యుత్ అధికారిని సంప్రదించండి {phone}"],
        },
        "legit": {
            "en": [
                "TGSPDCL: Bill for USC {usc} for Sep-2026 is Rs.{amt}. Due date {date}. Pay through the official app or website. Ignore if already paid.",
                "Dear consumer, power supply in {area} will be interrupted on {date} from 10:00 to 14:00 for maintenance work. Regret the inconvenience. -TGSPDCL",
            ],
            "hinglish": ["Aapka bijli bill USC {usc} ka Rs.{amt} hai, antim tithi {date}. Official app se bhugtan kare. -TGSPDCL"],
            "hi": ["आपका बिजली बिल (USC {usc}) Rs.{amt} है। अंतिम तिथि {date}। कृपया आधिकारिक ऐप से भुगतान करें।"],
            "te": ["మీ విద్యుత్ బిల్లు (USC {usc}) Rs.{amt}. చివరి తేదీ {date}. అధికారిక యాప్ ద్వారా చెల్లించండి. -TGSPDCL"],
        },
    },
    "parcel": {
        "scam": {
            "en": [
                "India Post: Your package is on hold at our warehouse due to incomplete address. Update within 24 hours or it will be returned: {scam_link}",
                "{courier}: Your parcel could not be delivered. Pay a redelivery fee of Rs.25 here {scam_link}",
                "This is {courier} customs. A parcel in your name containing illegal items has been seized. Call {phone} now to speak to the officer.",
            ],
            "hinglish": ["Aapka parcel address incomplete hone ki wajah se ruka hua hai. 24 ghante me address update kare warna parcel wapas chala jayega {scam_link}"],
            "hi": ["आपका पार्सल अधूरे पते के कारण रुका हुआ है। 24 घंटे में पता अपडेट करें: {scam_link}"],
            "te": ["మీ పార్సెల్ అసంపూర్ణ చిరునామా కారణంగా నిలిపివేయబడింది. 24 గంటల్లో చిరునామా అప్డేట్ చేయండి: {scam_link}"],
        },
        "legit": {
            "en": [
                "Your {courier} shipment {awb} is out for delivery today. Share OTP {otp} with the delivery agent only at the time of delivery.",
                "Delivered: your package {awb} was delivered to your doorstep at {time}. Track your orders on the {shop} app.",
            ],
            "hinglish": ["Aapka {courier} parcel {awb} aaj deliver hoga. Delivery ke samay hi agent ko OTP {otp} batayein."],
            "hi": ["आपका पार्सल {awb} आज डिलीवर होगा। डिलीवरी के समय ही एजेंट को OTP {otp} बताएं।"],
            "te": ["మీ పార్సెల్ {awb} ఈరోజు డెలివరీ అవుతుంది. డెలివరీ సమయంలో మాత్రమే ఏజెంట్‌కు OTP {otp} చెప్పండి."],
        },
    },
    "challan": {
        "scam": {
            "en": [
                "Your vehicle {vehicle} has an unpaid e-challan of Rs.{amt}. Pay today to avoid court action and licence suspension: {scam_link}",
                "RTO NOTICE: Traffic violation recorded for {vehicle}. Download the challan app to view details {scam_apk}",
            ],
            "hinglish": ["Aapki gaadi {vehicle} par Rs.{amt} ka challan baaki hai. Court case se bachne ke liye aaj hi bhugtan kare {scam_link}"],
            "hi": ["आपकी गाड़ी {vehicle} पर Rs.{amt} का चालान बकाया है। कोर्ट केस से बचने के लिए आज ही भुगतान करें {scam_link}"],
            "te": ["మీ వాహనం {vehicle} పై Rs.{amt} చలాన్ పెండింగ్‌లో ఉంది. కోర్టు కేసు నివారించడానికి ఈరోజే చెల్లించండి {scam_link}"],
        },
        "legit": {
            "en": [
                "e-Challan {ref} issued to vehicle {vehicle} for Rs.{amt} on {date}. Details and payment at https://echallan.parivahan.gov.in",
                "Payment of Rs.{amt} received for challan {ref}. Thank you. -Traffic Police",
            ],
            "hinglish": ["Gaadi {vehicle} ke liye challan {ref} jaari hua hai, Rs.{amt}. Vivaran ke liye https://echallan.parivahan.gov.in dekhein."],
            "hi": ["वाहन {vehicle} पर चालान {ref} जारी हुआ है, राशि Rs.{amt}। विवरण https://echallan.parivahan.gov.in पर देखें।"],
            "te": ["వాహనం {vehicle} కు చలాన్ {ref} జారీ చేయబడింది, Rs.{amt}. వివరాలకు https://echallan.parivahan.gov.in చూడండి."],
        },
    },
    "rewards": {
        "scam": {
            "en": [
                "Dear {bank} customer, your {pts} reward points worth Rs.{amt} expire today. Redeem now by installing: {scam_apk}",
                "Congratulations! You have received Rs.{amt} cashback on your {bank} credit card. Claim before midnight: {scam_link}",
            ],
            "hinglish": ["Aapke {bank} credit card ke {pts} reward points aaj expire ho rahe hai. Redeem karne ke liye app install kare {scam_apk}"],
            "hi": ["आपके {bank} क्रेडिट कार्ड के {pts} रिवॉर्ड पॉइंट्स आज समाप्त हो रहे हैं। अभी रिडीम करें {scam_link}"],
            "te": ["మీ {bank} క్రెడిట్ కార్డ్ {pts} రివార్డ్ పాయింట్లు ఈరోజు ముగుస్తాయి. ఇప్పుడే రిడీమ్ చేయండి {scam_apk}"],
        },
        "legit": {
            "en": [
                "You earned {pts} reward points on your {bank} credit card spend of Rs.{amt}. Check your points in the official app.",
                "Your {bank} credit card statement is generated. Total due Rs.{amt}, minimum due Rs.{amt2}, due date {date}.",
            ],
            "hinglish": ["Aapke {bank} card par Rs.{amt} kharch par {pts} reward points jude hai. Official app me dekhein."],
            "hi": ["आपके {bank} कार्ड पर Rs.{amt} खर्च करने पर {pts} रिवॉर्ड पॉइंट्स जुड़े हैं।"],
            "te": ["మీ {bank} కార్డ్‌పై Rs.{amt} ఖర్చుకు {pts} రివార్డ్ పాయింట్లు జమ అయ్యాయి."],
        },
    },
    "loan": {
        "scam": {
            "en": [
                "Pre-approved personal loan of Rs.{big} ready in 5 minutes. No documents, no CIBIL check. Pay processing fee Rs.{fee} to release: {scam_link}",
                "Your loan of Rs.{big} is approved. Share your Aadhaar and pay Rs.{fee} insurance charge to {upi} to get the money today.",
            ],
            "hinglish": ["Bina CIBIL check ke Rs.{big} ka loan 5 minute me. Sirf Rs.{fee} processing fee bhare {scam_link}"],
            "hi": ["बिना दस्तावेज के Rs.{big} का लोन 5 मिनट में। केवल Rs.{fee} प्रोसेसिंग फीस भरें {scam_link}"],
            "te": ["డాక్యుమెంట్లు లేకుండా 5 నిమిషాల్లో Rs.{big} లోన్. కేవలం Rs.{fee} ప్రాసెసింగ్ ఫీజు చెల్లించండి {scam_link}"],
        },
        "legit": {
            "en": [
                "Your EMI of Rs.{amt} for loan a/c {acct} is due on {date}. Please keep sufficient balance. -{bank}",
                "Dear customer, your personal loan application {ref} is under review. Our branch will contact you. We never charge any fee before disbursal.",
            ],
            "hinglish": ["Aapke loan a/c {acct} ki Rs.{amt} EMI {date} ko due hai. Kripya khate me paryapt balance rakhein. -{bank}"],
            "hi": ["आपके लोन खाते {acct} की Rs.{amt} की EMI {date} को देय है। कृपया पर्याप्त बैलेंस रखें।"],
            "te": ["మీ లోన్ ఖాతా {acct} EMI Rs.{amt} {date} న చెల్లించాలి. దయచేసి తగినంత బ్యాలెన్స్ ఉంచండి."],
        },
    },
    "upi": {
        "scam": {
            "en": [
                "Hello sir, I sent Rs.{amt} to your number by mistake. It was for my mother's hospital bill. Please send it back to {upi}, I am in trouble.",
                "You have received a payment request of Rs.{amt} to RECEIVE cashback. Enter your UPI PIN to accept the money.",
            ],
            "hinglish": ["Bhaiya galti se aapke number par Rs.{amt} chale gaye. Please wapas bhej do {upi} par, bahut zaroori hai."],
            "hi": ["भैया गलती से आपके नंबर पर Rs.{amt} चले गए। कृपया {upi} पर वापस भेज दें, बहुत ज़रूरी है।"],
            "te": ["అన్నా పొరపాటున మీ నంబర్‌కు Rs.{amt} పంపించాను. దయచేసి {upi} కి తిరిగి పంపండి, చాలా అవసరం."],
        },
        "legit": {
            "en": [
                "Rs.{amt} debited from A/c XX{acct4} to VPA {upi} on {date}. UPI Ref {ref}. Not you? Call your bank on the number printed on your card.",
                "Rs.{amt} credited to A/c XX{acct4} from {name} via UPI. Ref {ref}. -{bank}",
            ],
            "hinglish": ["Bro Rs.{amt} bhej diye GPay pe, dinner ka share. Check kar le."],
            "hi": ["आपके खाते XX{acct4} से Rs.{amt} UPI द्वारा {upi} को भेजे गए। संदर्भ {ref}।"],
            "te": ["మీ ఖాతా XX{acct4} నుండి Rs.{amt} UPI ద్వారా {upi} కి పంపబడింది. Ref {ref}."],
        },
    },
    "job": {
        "scam": {
            "en": [
                "Part time job! Earn Rs.{amt} daily by liking YouTube videos. Work from home, 10 minutes a day. WhatsApp HR: {scam_wa}",
                "Congratulations {name}, you are selected for an online task job. Daily salary Rs.{amt}. Join Telegram to start: {scam_tg}",
            ],
            "hinglish": ["Ghar baithe roz Rs.{amt} kamao, sirf YouTube videos like karke. Abhi join karo {scam_tg}"],
            "hi": ["घर बैठे रोज़ Rs.{amt} कमाएं, सिर्फ़ वीडियो लाइक करके। अभी जुड़ें {scam_tg}"],
            "te": ["ఇంటి నుండే రోజుకు Rs.{amt} సంపాదించండి, వీడియోలు లైక్ చేస్తే చాలు. ఇప్పుడే జాయిన్ అవ్వండి {scam_tg}"],
        },
        "legit": {
            "en": [
                "Hi {name}, your interview for the Software Engineer role is scheduled on {date} at {time}. Meeting link will be shared by email. -Talent Team",
                "Your application for the Data Analyst position has been viewed by the recruiter.",
            ],
            "hinglish": ["Hi {name}, kal {time} baje aapka interview hai. Resume ki copy saath laana. -HR"],
            "hi": ["नमस्ते {name}, आपका इंटरव्यू {date} को {time} बजे है। कृपया अपना रिज़्यूमे साथ लाएं।"],
            "te": ["హాయ్ {name}, మీ ఇంటర్వ్యూ {date} న {time} కి ఉంది. దయచేసి మీ రెజ్యూమ్ తీసుకురండి."],
        },
    },
    "family": {
        "scam": {
            "en": [
                "Hi Mum, this is my new number, my phone broke. Can you send Rs.{amt} urgently? I will explain later, don't call I can't talk.",
                "Papa I am in trouble, police caught me. Send Rs.{amt} to {upi} now please, don't tell anyone.",
            ],
            "hinglish": ["Mummy ye mera naya number hai, phone kharab ho gaya. Rs.{amt} jaldi bhejo please, baad me batata hoon. Call mat karna."],
            "hi": ["पापा मैं मुसीबत में हूं, जल्दी Rs.{amt} भेज दो इस नंबर पर, किसी को मत बताना।"],
            "te": ["అమ్మా ఇది నా కొత్త నంబర్, ఫోన్ పాడైంది. అర్జెంట్‌గా Rs.{amt} పంపించు, తర్వాత చెప్తా. కాల్ చేయకు."],
        },
        "legit": {
            "en": [
                "Hi Mum, reached the hostel safely. Will call you after dinner.",
                "Papa, I paid the college fee of Rs.{amt} today. Receipt is on WhatsApp.",
            ],
            "hinglish": ["Mummy hostel pahunch gaya, raat ko call karta hoon. Khana kha liya."],
            "hi": ["मम्मी मैं हॉस्टल पहुंच गया, रात को कॉल करूंगा।"],
            "te": ["అమ్మా హాస్టల్‌కి చేరుకున్నా, రాత్రి కాల్ చేస్తా."],
        },
    },
    "otp": {
        "scam": {
            "en": [
                "Your {bank} account will be credited with Rs.{amt} refund. Share the OTP you just received with our executive to complete the process.",
                "This is {bank} customer care. To stop the unauthorised transaction of Rs.{amt}, please tell us the OTP sent to your mobile.",
            ],
            "hinglish": ["Sir aapke account me Rs.{amt} refund aana hai. Abhi jo OTP aaya hai woh bata dijiye process complete karne ke liye."],
            "hi": ["आपके खाते में Rs.{amt} रिफंड आना है। अभी आया OTP हमारे अधिकारी को बताएं।"],
            "te": ["మీ ఖాతాలో Rs.{amt} రిఫండ్ రావాలి. ఇప్పుడు వచ్చిన OTP మా ఎగ్జిక్యూటివ్‌కు చెప్పండి."],
        },
        "legit": {
            "en": [
                "{otp} is your OTP for a transaction of Rs.{amt} at {shop}. Valid for 5 minutes. Do not share this OTP with anyone. -{bank}",
                "{otp} is your login OTP for {shop}. Never share it with anyone, including our staff.",
            ],
            "hinglish": ["{otp} aapka OTP hai Rs.{amt} ke transaction ke liye. Yeh OTP kisi ke saath share na kare. -{bank}"],
            "hi": ["{otp} आपका OTP है Rs.{amt} के लेनदेन के लिए। यह OTP किसी के साथ साझा न करें।"],
            "te": ["Rs.{amt} లావాదేవీకి మీ OTP {otp}. ఈ OTP ఎవరితోనూ పంచుకోవద్దు. -{bank}"],
        },
    },
    # --- scam-only topics -------------------------------------------------------------
    "digital_arrest": {
        "scam": {
            "en": [
                "This is CBI Mumbai. Your Aadhaar number is linked to a money laundering case. You are under digital arrest. Do not disconnect and do not inform family. Call {phone}.",
                "Notice from Cyber Crime Cell: an FIR is registered against your mobile number for illegal activity. Join the video call with the officer within 2 hours or face arrest.",
            ],
            "hinglish": ["Aapke Aadhaar se money laundering case juda hai. Aap digital arrest me hai. Kisi ko mat batana, turant {phone} par call kare."],
            "hi": ["आपके आधार नंबर से मनी लॉन्ड्रिंग केस जुड़ा है। आप डिजिटल अरेस्ट में हैं। किसी को न बताएं।"],
            "te": ["మీ ఆధార్ నంబర్ మనీ లాండరింగ్ కేసుతో లింక్ అయింది. మీరు డిజిటల్ అరెస్ట్‌లో ఉన్నారు. ఎవరికీ చెప్పకండి."],
        },
    },
    "lottery": {
        "scam": {
            "en": [
                "Congratulations! Your number has won Rs.25,00,000 in the KBC lucky draw. To claim, contact manager {name} on WhatsApp {phone}.",
                "You have won an iPhone 17 in the {shop} anniversary lucky draw. Pay Rs.{fee} delivery charge to receive it: {scam_link}",
            ],
            "hinglish": ["Badhai ho! Aapka number KBC lucky draw me Rs.25 lakh jeeta hai. Claim ke liye {phone} par WhatsApp kare."],
            "hi": ["बधाई हो! आपके नंबर ने KBC लकी ड्रॉ में 25 लाख रुपये जीते हैं। दावा करने के लिए {phone} पर संपर्क करें।"],
            "te": ["అభినందనలు! మీ నంబర్ KBC లక్కీ డ్రాలో 25 లక్షలు గెలుచుకుంది. క్లెయిమ్ చేయడానికి {phone} కి వాట్సాప్ చేయండి."],
        },
    },
    "investment": {
        "scam": {
            "en": [
                "Join our VIP stock group. Guaranteed 300% returns in 30 days on IPO allotment. Limited seats: {scam_wa}",
                "Dear investor, our trading app gives 5% daily profit. Deposit Rs.{amt} today and withdraw anytime: {scam_link}",
            ],
            "hinglish": ["Stock market me 30 din me paisa teen guna. VIP group join karo abhi {scam_wa}"],
            "hi": ["शेयर बाज़ार में 30 दिन में पैसा तीन गुना। अभी VIP ग्रुप से जुड़ें {scam_wa}"],
            "te": ["స్టాక్ మార్కెట్‌లో 30 రోజుల్లో మీ డబ్బు మూడు రెట్లు. ఇప్పుడే VIP గ్రూప్‌లో చేరండి {scam_wa}"],
        },
    },
    # --- legit-only topics ------------------------------------------------------------
    "personal": {
        "legit": {
            "en": [
                "Bro where are you? Match starts at {time}, bring the bat.",
                "Meeting moved to {time} tomorrow, same room. Please share the slides before that.",
            ],
            "hinglish": ["Kal {time} baje milte hai metro station pe, late mat hona."],
            "hi": ["कल {time} बजे मेट्रो स्टेशन पर मिलते हैं।"],
            "te": ["రేపు {time} కి మెట్రో స్టేషన్ దగ్గర కలుద్దాం."],
        },
    },
    "transactional": {
        "legit": {
            "en": [
                "Salary of Rs.{big} credited to your A/c XX{acct4} on {date}. -{bank}",
                "Recharge of Rs.{amt} successful for {phone}. Validity 28 days.",
            ],
            "hinglish": ["Aapka Rs.{amt} ka recharge safal raha. Validity 28 din."],
            "hi": ["आपके खाते XX{acct4} में Rs.{big} वेतन जमा किया गया।"],
            "te": ["మీ ఖాతా XX{acct4} లో Rs.{big} జీతం జమ అయింది."],
        },
    },
    "reminders": {
        "legit": {
            "en": [
                "Your LPG cylinder booking {ref} is confirmed. Delivery expected by {date}.",
                "Reminder: parent-teacher meeting on {date} at {time} in the school auditorium.",
            ],
            "hinglish": ["Aapki gas booking {ref} confirm ho gayi hai. Delivery {date} tak."],
            "hi": ["आपकी गैस बुकिंग {ref} कन्फर्म हो गई है। डिलीवरी {date} तक।"],
            "te": ["మీ గ్యాస్ బుకింగ్ {ref} కన్ఫర్మ్ అయింది. డెలివరీ {date} లోపు."],
        },
    },
}

LANGS = ("en", "hinglish", "hi", "te")


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _code(rng: random.Random, n: int = 7) -> str:
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(n))


def _ip(rng: random.Random) -> str:
    return f"{rng.randint(11, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _scam_link(rng: random.Random, b: str) -> str:
    tld = rng.choice(SCAM_TLDS)
    k = rng.randrange(7)
    if k == 0:
        return f"http://{b}-kyc-update.{tld}/verify"
    if k == 1:
        return f"https://{b}.secure-login.{tld}/account"
    if k == 2:
        return f"https://bit.ly/{_code(rng)}"
    if k == 3:
        return f"http://{_ip(rng)}/{b}/login"
    if k == 4:
        return f"https://{b}-netbanking-in.{tld}"
    if k == 5:
        return f"https://{LOOKALIKE[b]}/login"
    return f"https://www.{rng.choice(SCAM_SITES)}.{tld}/pay"


def _scam_apk(rng: random.Random, b: str) -> str:
    name = rng.choice(["Rewards", "Update", "Challan_Report", "KYC"])
    if rng.random() < 0.3:
        return f"http://{_ip(rng)}/app/{name}.apk"
    return f"https://{b}-{rng.choice(['yono', 'rewards', 'update', 'challan'])}.{rng.choice(SCAM_TLDS)}/{b.upper()}_{name}.apk"


def _amount(rng: random.Random, lo: int, hi: int) -> str:
    v = rng.randint(lo, hi)
    return f"{v:,}" if rng.random() < 0.5 else str(v)


def _slots(rng: random.Random) -> dict[str, str]:
    bank, btok, _ = rng.choice(BANKS)
    name = rng.choice(NAMES)
    phone = f"+91 {rng.choice('6789')}{_digits(rng, 9)}"
    return {
        "bank": bank,
        "name": name,
        "area": rng.choice(AREAS),
        "shop": rng.choice(SHOPS),
        "courier": rng.choice(COURIERS),
        "phone": phone,
        "upi": f"{rng.choice(NAMES).lower()}{rng.randint(1, 999)}@{rng.choice(['ybl', 'okaxis', 'oksbi', 'paytm', 'ibl'])}",
        "scam_link": _scam_link(rng, btok),
        "scam_apk": _scam_apk(rng, btok),
        "scam_wa": f"https://wa.me/91{rng.choice('6789')}{_digits(rng, 9)}",
        "scam_tg": f"https://t.me/+{_code(rng, 10)}",
        "amt": _amount(rng, 99, 9999),
        "amt2": _amount(rng, 99, 999),
        "big": _amount(rng, 50_000, 500_000),
        "fee": _amount(rng, 499, 4999),
        "pts": str(rng.randint(500, 9000)),
        "otp": _digits(rng, 6),
        "ref": _digits(rng, rng.choice([10, 12])),
        "awb": rng.choice([f"EK{_digits(rng, 9)}IN", _digits(rng, 11)]),
        "acct": _digits(rng, 12),
        "acct4": _digits(rng, 4),
        "date": f"{rng.randint(1, 28):02d}-{rng.randint(9, 12):02d}-2026",
        "time": rng.choice([f"{rng.randint(7, 11)}:{rng.choice(['00', '15', '30', '45'])} PM",
                            f"{rng.randint(18, 23)}:{rng.choice(['00', '30'])}"]),
        "usc": _digits(rng, 9),
        "vehicle": f"{rng.choice(['TS', 'TG'])}{rng.randint(1, 38):02d}{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}{_digits(rng, 4)}",
    }


def build_corpus(fills: int = 6, seed: int = 0) -> list[dict]:
    """Render every template ``fills`` times. Returns one dict per message."""
    rng = random.Random(seed)
    rows = []
    for topic, by_label in TEMPLATES.items():
        for label_name, by_lang in by_label.items():
            for lang, templates in by_lang.items():
                for k, tpl in enumerate(templates):
                    tid = f"{topic}.{label_name}.{lang}.{k}"
                    for _ in range(fills):
                        rows.append({
                            "text": tpl.format(**_slots(rng)),
                            "label": int(label_name == "scam"),
                            "topic": topic,
                            "lang": lang,
                            "template_id": tid,
                        })
    return rows
