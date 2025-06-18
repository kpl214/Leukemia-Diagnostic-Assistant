<template>
  <div class="chat-container">
    <h2>Leukemia Diagnostic Assistant</h2>

    <div class="chat-box">
      <div v-for="(msg, index) in messages" :key="index" :class="['message', msg.sender]">
        <div class="bubble">
          <p>{{ msg.text }}</p>
        </div>
      </div>
    </div>

    <div class="input-area">
      <textarea
        v-model="userInput"
        placeholder="Describe your patient case or ask a question..."
        rows="2"
      ></textarea>
      <button @click="sendMessage">Send</button>
    </div>

    <div class="upload-section">
      <input type="file" @change="uploadImage" />
      <div v-if="imageResult" class="image-result">
        <strong>🩸 Image Classification:</strong> {{ imageResult }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const userInput = ref('')
const messages = ref([])
const imageResult = ref('')

const sendMessage = async () => {
  if (!userInput.value.trim()) return

  messages.value.push({ sender: 'user', text: userInput.value })

  try {
    const res = await axios.post('http://localhost:5000/api/chat', { prompt: userInput.value })
    messages.value.push({ sender: 'bot', text: res.data.response })
  } catch (err) {
    messages.value.push({ sender: 'bot', text: '❌ Failed to get response from server.' })
  }

  userInput.value = ''
}

const uploadImage = async (e) => {
  const file = e.target.files[0]
  if (!file) return

  const formData = new FormData()
  formData.append('image', file)

  try {
    const res = await axios.post('http://localhost:5000/api/upload', formData)
    imageResult.value = res.data.classification
    messages.value.push({ sender: 'bot', text: `🩸 Image classified as: ${res.data.classification}` })
  } catch (err) {
    messages.value.push({ sender: 'bot', text: '❌ Image classification failed.' })
  }
}
</script>

<style scoped>
.chat-container {
  max-width: 700px;
  margin: 40px auto;
  font-family: Arial, sans-serif;
}

h2 {
  text-align: center;
}

.chat-box {
  border: 1px solid #ccc;
  padding: 1rem;
  height: 400px;
  overflow-y: auto;
  background: #D3D3D3;
  border-radius: 10px;
  margin-bottom: 1rem;
}

.message {
  margin-bottom: 0.8rem;
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.message.bot {
  justify-content: flex-start;
}

.bubble {
  padding: 10px 15px;
  border-radius: 15px;
  max-width: 75%;
  white-space: pre-wrap;
  text-align: left;
  align-self: flex-start;
}

.user .bubble {
  background-color: #7373FF;
  color: white;
  box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
}

.bot .bubble {
  background-color: #f0f0f0;
  color: black;
  box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
}

.input-area {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
}

textarea {
  flex: 1;
  resize: none;
  padding: 0.5rem;
}

button {
  padding: 0.5rem 1rem;
  cursor: pointer;
}

.upload-section {
  margin-top: 1rem;
  text-align: center;
}

.image-result {
  margin-top: 0.5rem;
  background: #535bf2;
  padding: 0.5rem;
  border-radius: 8px;
}
</style>
