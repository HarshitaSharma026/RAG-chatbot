import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads
from operator import itemgetter
import streamlit as st
load_dotenv()


load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

persist_directory = "../vectordb"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma(
  persist_directory=persist_directory,
  embedding_function = embeddings
)

retriever = vectorstore.as_retriever()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


template = """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from the vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}"""

# Correct implementation of multiquery_prompt
multiquery_prompt = PromptTemplate(
  template=template,
  input_variables=['question']
)

multiquery_chain = (
  multiquery_prompt
  | model
  | StrOutputParser()
  | (lambda x: [q.strip() for q in x.split('\n') if q.strip()])
)

def reciprocal_rank_fusion(results: list[list], k = 60):
  rrf_scores = {}
  for docs in results:
    for rank, doc in enumerate(docs):
      doc_str = dumps(doc)
      if doc_str not in rrf_scores:
        rrf_scores[doc_str] = 0
      prev_score = rrf_scores[doc_str]
      rrf_scores[doc_str] = 1 / (rank + k)
  reranked_scores = [
    (loads(doc), score)
    for doc, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
  ]
  return reranked_scores


st.header('I am here to help you. Ask me anything.')
st.title('Enhanced RAG chatbot')

# defining history if its not already there
if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

# display all messages on the screen
for message in st.session_state.chat_history:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

user_input = st.chat_input('Ask me anything.')
if user_input:
  # display it on screen
  with st.chat_message('user'):
    st.markdown(user_input)

  # append user input to history
  st.session_state.chat_history.append({'role': 'user', 'content': user_input})

  # display chatbot output
  with st.chat_message('assistant'):
    doc_chain = multiquery_chain | retriever.map() | reciprocal_rank_fusion
    # -------------- few shot prompting
    rag_template = """You are an AI assistant answering questions based on the provided context. Here are some examples of how you should respond:\n
    Example 1: Query about academic calender\n
    Context: \nAcademic Calendar(2024 - 2025)Winter Semester
    Academic Calendar for Winter Semester 2024 - 25 (Applicable to the students of all the programmes except BSc. (Agri) & MBA)
    date: 09.10.2024 & 10.10.2024 
    day: Wednesday & Thursday 
    description: Course wish list registration by students
    date: 14.10.2024 - 29.10.2024 
    day: Monday to Tuesday 
    description: Course allocation and scheduling by Schools 
    date: 07.11.2024 
    day: Thursday 
    description: Mock - Course registration for Freshers
    date: 09.11.2024 
    day: Saturday 
    description: Course registration by students
    date: 13.12.2024 
    day: Friday 
    description: Commencement of Winter Semester 2024-25
    date: 13.12.2024-15.12.2024 
    day: Friday to Sunday 
    description: Course add/drop option for students
    date: 22.12.2024-01.01.2025 
    day: Sunday to Wednesday 
    description: Winter Vacation for the students (11 Days)
    date: 05.01.2025 
    day: Sunday 
    description: Last date for the payment of re-registration fees
    date: 14.01.2025 
    day: Tuesday 
    description: Pongal (Holiday)
    date: 27.01.2025-02.02.2025 
    day: Monday to Sunday 
    description: Continuous Assessment Test -1
    date: 20.02.2025-23.02.2025 
    day: Thursday to Sunday 
    description: Riviera 2025
    date: 24.02.2025-26.02.2025 
    day: Monday to Wednesday 
    description: Course withdraw option for students
    date: 14.03.2025 
    day: Friday 
    description: Holi (Holiday)
    date: 16.03.2025-22.03.2025 
    day: Sunday to Saturday 
    description: Continuous Assessment Test - II
    date: 30.03.2025 
    day: Sunday 
    description: Telugu New Year's Day
    date: 31.03.2025 
    day: Monday 
    description: Ramzan (No Instructional Day)
    date: 05.04.2025 
    day: Saturday 
    description: Last instructional day for laboratory classes
    date: 07.04.2025-11.04.2025 
    day: Monday to Friday 
    description: Final assessment test for laboratory courses/components
    date: 14.04.2025 
    day: Monday 
    description: Tamil New Year's Day/ Dr. B. R. Ambedkar Birthday(Holiday)
    date: 17.04.2025 
    day: Thursday 
    description: Last instructional day for theory classes
    date: 18.04.2025 
    day: Friday 
    description: Good Friday (No Instructional Day)
    date: 21.04.2025 
    day: Monday 
    description: Commencement of final assessment test for theorycourses / components
    date: 12.05.2025 
    day: Monday 
    description: Commencement of Summer Term 2024-25 (Tentative)
    date: 14.07.2025 
    day: Monday 
    description: Commencement of Fall Semester 2025-26 (Tentative) 
    Question: last working day of winter semester 2025?
    Answer: The last intructional day for theory classes is 18.04.2025 (friday) and lab classes as (07.04.2025-11.04.2025).\n\n

    Example 2: Syllabus related queries. Show answer in proper format.\n
    Context: \n
    Syllabus Advanced Database Management Systems(5year_cse)
    Course code: CSI2004
    Course title:Advanced Database Management Systems  
    Module:1 Database Design Techniques 5 hours Review of DBMS Techniques – EER – Physical database design and tuning – Advanced transaction processing and Query processing Module:2 Parallel Databases 6 hours Architecture, Data partitioning strategy, Interquery and Intraquery Parallelism –Parallel query optimization Module:3 Distributed Databases 7 hours Structure of distributed database, Advantages, Functions, Distributed database architecture, Allocation, Fragmentation, Replication, Distributed query processing, Distributed transaction processing, Concurrency control and Recovery in distributed database systems. Module:4 Multimedia and Spatial Databases 7 hours Multimedia sources, issues, Multimedia database applications Multimedia database queries -LOB in SQL. Spatial databases -Type of spatial data – Indexing in spatial databases. Module:5 Mobile and Cloud Databases 8 hours 1. Wireless network communication, Location and handoff management, Data processing and mobility, Transaction management in mobile database systems, Database options in the cloud , Changing role of the DBA in the cloud , Moving your databases to the cloud Module:6 Emerging Database Technologies 5 hours Active database – Detective database - Object database - Temporal database - Streaming databases Module:7 Database Security 5 hours Introduction to Database Security Issues –Security Models – Different Threats to databases – Counter measures to deal with these problems Module:8 Recent Trends 2 hours Total Lecture hours: 45 hours Text Book(s) 1. Raghu Ramakrishnan , Database Management Systems , ,4th edition, Mcgraw -Hill,2015 2. Abraham Silberschatz, Henry F. Korth, S. Sudharshan, “Database System Concepts”, Seventh Edition, Tata McGraw Hill, 2019. Reference Books 1. RamezElmasri, Shamkant B. Navathe, “Fundamentals of Database Systems”, Seventh Edition, Pearson Education, 2016. 2. Vlad Vlasceanu, Wendy A. Neu, Andy Oram, Sam Alapati, “An Introduction to Cloud Databases”, O'Reilly Media, Inc. 2019 3. S.K.Singh, Database Systems: Concepts, Design & Applications, 2nd Edition, Pearson education, 2011 Mode of Evaluation: CAT/ Digital Assignments/ Quiz/ FAT/ Project.
    Syllabus Principles of Compiler Design(5year_cse)
    Course code: CSI2004
    Course title: Principles of Compiler Design 
    Module:1 Introduction to Compilation and Lexcial Analysis 7 hours Introduction to programming language translators -Structure and phases of a compiler -Design issues - Patterns - lexemes -Tokens -Attributes -Specification of Tokens - Extended Regular expression, Regular expression to Deterministic Finite Automata (Direct method). Module:2 Syntax Analysis –Top Down 5 hours Role of parser - Parse Tree - Elimination of ambiguity - Top down parsing - Recursive Descent parsing - Non Recursive Descent parsing - Predictive Parsing - LL(1) grammars. Module:3 Syntax Analysis –Bottom Up 7 hours Shift Reduce Parsers - Operator Precedence Parsing ,LR parsers: -Construction of SLR parser tables and parsing , CLR parsing -LALR parsing Module:4 Semantics Analysis 6 hours Syntax Directed Definition – Evaluation Order - Applications of Syntax Directed Translation - Syntax Directed Translation Schemes - Implementation of L attributed Syntax Directed Definition. Module:5 Intermediate Code Generation 7 hours Variants of syntax trees - Three address code - Types – Declarations - Procedures - Assignment Statements - Translation of Expressions - Control Flow - Back Patching - Switch Case Statements. Module:6 Code Optimization 6 hours Loop optimizations - Principal sources of optimization -Introduction to Data Flow Analysis - Basic Blocks - The DAG Representation of Basic Blocks -Loops in Flow Graphs. Module:7 Code Generation & Other Translations Issues 5 hours Issues in the design of a code generator - Target Machine - Next -Use Information - Optimization of basic blocks - Peephole Optimization - Register Allocation and Assignment. Module:8 Recent Trends 2 hours Total Lecture hours: 45 hours Text Book(s) A. V. Aho, Monica S. Lam, Ravi Sethi and Jeffrey D. Ullman, Compilers: Principles, Techniques, & Tools, Second Edition, , Pearson Education, 2007 K. D. Cooper and L. Torczon, Engineering a Compiler, 2nd e dition. Morgan Kaufmann,2011. Reference Books Andrew A.Appel , Modern Compiler Implementation in Java, 2nd edition, Cambridge University Press;, 2002. Allen Holub, Compiler Design in C, Prentice Hall,1990. Torbengidius Mogensen, “Basics of Compiler Design”, Springer, 2011. Mode of Evaluation: CAT / Assignment / Quiz / FAT / Project / Seminar
    Question: syllabus of Principles of Compiler Design in mtech cse.
    Answer: Syllabus of Principles of Compiler Design (Mtech CSE)
    Course code: CSI2004
    Course title: Principles of Compiler Design 
    Module:1 Introduction to Compilation and Lexcial Analysis 7 hours Introduction to programming language translators -Structure and phases of a compiler -Design issues - Patterns - lexemes -Tokens -Attributes -Specification of Tokens - Extended Regular expression, Regular expression to Deterministic Finite Automata (Direct method). Module:2 Syntax Analysis –Top Down 5 hours Role of parser - Parse Tree - Elimination of ambiguity - Top down parsing - Recursive Descent parsing - Non Recursive Descent parsing - Predictive Parsing - LL(1) grammars. Module:3 Syntax Analysis –Bottom Up 7 hours Shift Reduce Parsers - Operator Precedence Parsing ,LR parsers: -Construction of SLR parser tables and parsing , CLR parsing -LALR parsing Module:4 Semantics Analysis 6 hours Syntax Directed Definition – Evaluation Order - Applications of Syntax Directed Translation - Syntax Directed Translation Schemes - Implementation of L attributed Syntax Directed Definition. Module:5 Intermediate Code Generation 7 hours Variants of syntax trees - Three address code - Types – Declarations - Procedures - Assignment Statements - Translation of Expressions - Control Flow - Back Patching - Switch Case Statements. Module:6 Code Optimization 6 hours Loop optimizations - Principal sources of optimization -Introduction to Data Flow Analysis - Basic Blocks - The DAG Representation of Basic Blocks -Loops in Flow Graphs. Module:7 Code Generation & Other Translations Issues 5 hours Issues in the design of a code generator - Target Machine - Next -Use Information - Optimization of basic blocks - Peephole Optimization - Register Allocation and Assignment. Module:8 Recent Trends 2 hours Total Lecture hours: 45 hours Text Book(s) A. V. Aho, Monica S. Lam, Ravi Sethi and Jeffrey D. Ullman, Compilers: Principles, Techniques, & Tools, Second Edition, , Pearson Education, 2007 K. D. Cooper and L. Torczon, Engineering a Compiler, 2nd e dition. Morgan Kaufmann,2011. Reference Books Andrew A.Appel , Modern Compiler Implementation in Java, 2nd edition, Cambridge University Press;, 2002. Allen Holub, Compiler Design in C, Prentice Hall,1990. Torbengidius Mogensen, “Basics of Compiler Design”, Springer, 2011. Mode of Evaluation: CAT / Assignment / Quiz / FAT / Project / Seminar. \n\n

    Example 3: Faculty related queries
    Context:\n
    10873
    Dr. Senthil Kumar P
    Machine Learning, Big Data Analytics, Deep Learning
    9994626135
    psenthilkumar@vit.ac.in
    TUESDAY, 11:30 AM - 12:30 PM, THURSDAY, 03:00 PM - 04:00 PM
    SJT-210-A13
    11627
    Dr. Senthil Murugan B
    Cloud Computing, Big Data Analytics, Future Internet (Internet of Things, Internet of Services , Internet of Content), AI & Machine\nLearning, Data Engineering
    9047151090
    senthilmurugan.b@vit.ac.in
    WEDNESDAY, 10:00 AM - 11:00 AM, FRIDAY, 12:00 PM - 01:00 PM
    SJT-411-A20
    12340
    Dr. Shynu P G
    Cloud computing, Machine Learning,  Big Data Analytics, IoT, Block Chain Technology
    9840800396
    pgshynu@vit.ac.in
    No information
    SJT-116-A02
    12388
    Dr. Srinivas Koppu
    Blockchain, IoT, Federated Learning, AI,  Data Analytics
    7667163460
    srinukoppu@vit.ac.in
    THURSDAY, 10:00 AM - 11:00 AM, WEDNESDAY, 11:15 AM - 12:45 PM, TUESDAY, 11:00 AM - 12:15 PM
    SJT-111-A12
    Question: information of shynu sir.
    Answer: \n
    Emp ID: 12340
    Faculty Name: Dr. Shynu P G
    Area of specialization: Cloud computing, Machine Learning,  Big Data Analytics, IoT, Block Chain Technology
    Mobile No.: 9840800396
    E-mail ID: pgshynu@vit.ac.in
    Open Hours:  No information provided
    Cabin Number: SJT-116-A02 \n\n

    Example 4: Answering faculty related information. In cases where there are more than one faculty or professor with the same name, get the required information of all the faculty having same name if full name is not mentioned properly. Ask for follow-up question if required but also provide details of all faculty or professor having same name. \n
    Context:\n
    14795
    Dr. Ramkumar T
    Big Data Analytics, Data Mining
    9442421674
    ramkumar.thirunavukarasu@vit.ac.in
    MONDAY, 09:30 AM - 11:30 AM, TUESDAY, 08:30 AM - 09:45 AM, WEDNESDAY, 10:00 AM - 12:00 PM, THURSDAY, 10:00 AM - 12:00 PM, FRIDAY, 12:00 PM - 01:00 PM
    SJT-210-A39
    11160
    Dr. Senthil Kumaran  U
    Wireless and Mobile, Data Mining, Software Engineering, Cloud Computing
    9994863891
    usenthilkumaran@vit.ac.in
    No information
    SJT-313-A01
    12295
    Dr. Senthil kumar M
    Machine Learning,  Deep Learning,  Big Data, Security
    9994267718
    senthilkumar.mohan@vit.ac.in
    WEDNESDAY, 10:30 AM - 12:30 AM, FRIDAY, 10:30 AM - 12:30 PM
    SJT-116-A11
    10873
    Dr. Senthil Kumar P
    Machine Learning, Big Data Analytics, Deep Learning
    9994626135
    psenthilkumar@vit.ac.in
    TUESDAY, 11:30 AM - 12:30 PM, THURSDAY, 03:00 PM - 04:00 PM
    SJT-210-A13
    11627
    Dr. Senthil Murugan B
    Cloud Computing, Big Data Analytics, Future Internet (Internet of Things, Internet of Services , Internet of Content), AI & Machine\nLearning, Data Engineering
    9047151090
    senthilmurugan.b@vit.ac.in
    WEDNESDAY, 10:00 AM - 11:00 AM, FRIDAY, 12:00 PM - 01:00 PM
    SJT-411-A20
    11599
    Dr. Seetha R
    Mobile ad hoc network, Cryptography, Blockchain, IoT
    9486341417
    rseetha@vit.ac.in
    MONDAY, 12:00 PM - 01:00 PM, THURSDAY, 04:00 PM - 05:00 PM
    SJT-213-A32
    10364
    Dr. Senthilkumar N.C
    Data Mining, Machine Learning, Deep Learning, Big Data Analytics, Sentiment Analysis. Opinion mining, Text mining, IoT, Image\nProcessing
    9486845866
    ncsenthilkumar@vit.ac.in
    No information
    SJT-313-A35
    12374
    Dr. Ramya.G
    Formal languages and automata theory Machine learning, Artificial intelligence
    9597218281
    ramya.g@vit.ac.in
    THURSDAY, 01:00 PM - 05:00 PM\nMONDAY, 10:00 AM - 01:00 PM
    SJT-116-A38
    11581
    Dr. Senthilkumar N
    Semantic Web, Information Retrieval and Natural Language Processing
    9943541777
    senthilkumar.n@vit.ac.in
    MONDAY, 11:00 AM - 12:00 PM\nTHURSDAY, 02:00 PM - 03:00 PM
    SJT-310-A22
    19610
    Dr. Senthilkumar T
    Image Processing, Networking, Wireless & Ad hOC Networks
    9442254807
    senthilkumar.t@vit.ac.in
    No information
    PRP BLOCK 218 L
    Question: cabin number and open hours of sentil sir.
    Answer:\n
    There are more than one sentil sir. The cabin number and open hours of all are provided below.
    Faculty Name: Dr. Senthilkumar T
    Open Hours:  No information provided
    Cabin Number: PRP BLOCK 218 L
    Faculty Name: Dr. Senthilkumar N
    Open Hours:  MONDAY, 11:00 AM - 12:00 PM\nTHURSDAY, 02:00 PM - 03:00 PM
    Cabin Number: SJT-310-A22
    Faculty Name: Dr. Senthilkumar N.C
    Open Hours:  No information provided
    Cabin Number: SJT-313-A35
    Faculty Name: Dr. Senthil Murugan B
    Open Hours:  WEDNESDAY, 10:00 AM - 11:00 AM, FRIDAY, 12:00 PM - 01:00 PM
    Cabin Number: SJT-411-A20
    Faculty Name: Dr. Senthilkumar N.C
    Open Hours:  No information provided
    Cabin Number: SJT-313-A35
    Faculty Name: Dr. Senthil Murugan B
    Open Hours:  WEDNESDAY, 10:00 AM - 11:00 AM, FRIDAY, 12:00 PM - 01:00 PM
    Cabin Number: SJT-411-A20
    Faculty Name: Dr. Senthil Kumaran  U
    Open Hours: No information
    Cabin Number:SJT-313-A01
    Faculty Name: Dr. Senthil kumar M
    Open Hours:WEDNESDAY, 10:30 AM - 12:30 AM, FRIDAY, 10:30 AM - 12:30 PM
    Cabin Number:SJT-116-A11
    Faculty Name: Dr. Senthil Kumar P
    Open Hours:TUESDAY, 11:30 AM - 12:30 PM, THURSDAY, 03:00 PM - 04:00 PM
    Cabin Number:SJT-210-A13\n\n

    Do not mention explicitly that you're providing the answer from the given context by saying "The provided text mentions" or "Based on provided text". Just directly give the appropriate answer as if the user thinks that you have this knowledge inbuilt in you. If you are unable to frame any answer, do not make it up, and simply say "Unable to answer this question at this moment." If someone asks who are you, answer response that you're assistant designed to help students of VIT. If someone asks who is your developer or who built you, answer a student from VIT only. Give your responses in proper format, and use pointers wherever needed.
    Now, answer the following question based on this context:
    Context: {context}
    Question: {question}"""
      # Correct implementation of rag_prompt
    rag_prompt = ChatPromptTemplate.from_messages([
      ("system", rag_template),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{question}")
    ])

    chain = (
      {'context': doc_chain, 'question': itemgetter('question'), 'chat_history': itemgetter('chat_history')}
      | rag_prompt
      | model
      | StrOutputParser()
    )
    result = chain.invoke({'question': user_input, 'chat_history': st.session_state.chat_history})

    st.markdown(result)
    st.session_state.chat_history.append({'role': 'assistant', 'content': result})
