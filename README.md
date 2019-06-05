# AndrOpGAN

Source Code of AndrOpGAN: An Android Opcode Generative Adversarial Network for the Obfuscation in Making Malwares

---
This project contains the source code for APK opcode distribution feature generation module and feature insertion module.

---
### Requirements
- Tensorflow
- tflearn
- Apktool

---
### Program description
- **Generator_DCGAN.py**

Train the generator, save the model, and generate opcode distribution feature data.

- **get_opcode_sequence_file.py**

Extract opcode sequence files from APK.

- **Insertion_module.py**

Feature Library Data Processing, Insert Opcode, Rebuild APK.

- **sequence_file_process.py**

Convert opcode sequence files to distribution features.

---


### Code reference：
Part of the code refers to the following procedures, thank you very much for their selfless open source
- https://github.com/niallmcl/Deep-Android-Malware-Detection/blob/master/opcodeseq_creator/run_opcode_seq_creation.py
- https://github.com/tflearn/tflearn/blob/master/examples/images/dcgan.py
- Apktool
