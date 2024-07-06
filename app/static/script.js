document.body.addEventListener('htmx:beforeRequest', function(event) {
    var message = document.getElementById('message-input').value;
    var chatContainer = document.getElementById('chat-container');
    
    // Append user message immediately
    var userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.innerHTML = `<p>${message}</p>`;
    chatContainer.appendChild(userMessage);
    
    // Trigger reflow to ensure the transition happens
    userMessage.offsetHeight;
    
    // Add 'show' class to trigger animation
    userMessage.classList.add('show');
    
    // Clear input field
    document.getElementById('message-input').value = '';
    
    // Append typing indicator
    var typingIndicator = document.getElementById('typing-indicator').content.cloneNode(true);
    chatContainer.appendChild(typingIndicator);
});

document.body.addEventListener('htmx:afterSwap', function(event) {
    var chatContainer = document.getElementById('chat-container');
    
    // Remove typing indicator
    var typingIndicator = chatContainer.querySelector('.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
    
    // Get the newly added bot message
    var newMessage = chatContainer.lastElementChild;
    
    // Trigger reflow to ensure the transition happens
    newMessage.offsetHeight;
    
    // Add 'show' class to trigger animation
    newMessage.classList.add('show');
});
