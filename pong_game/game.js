// Pong Game with AI Opponent
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const playerScoreElement = document.getElementById('playerScore');
const aiScoreElement = document.getElementById('aiScore');
const startBtn = document.getElementById('startBtn');
const resetBtn = document.getElementById('resetBtn');
const difficultyBtn = document.getElementById('difficultyBtn');

// Game settings
const paddleWidth = 15;
const paddleHeight = 100;
const ballSize = 12;
let playerScore = 0;
let aiScore = 0;
let difficulty = 'easy'; // easy, medium, hard

// Game objects
const playerPaddle = {
    x: 20,
    y: canvas.height / 2 - paddleHeight / 2,
    width: paddleWidth,
    height: paddleHeight,
    speed: 8,
    dy: 0
};

const aiPaddle = {
    x: canvas.width - 20 - paddleWidth,
    y: canvas.height / 2 - paddleHeight / 2,
    width: paddleWidth,
    height: paddleHeight,
    speed: 0, // Will be set based on difficulty
    dy: 0
};

const ball = {
    x: canvas.width / 2,
    y: canvas.height / 2,
    size: ballSize,
    speed: 5,
    dx: 5,
    dy: 5
};

// Game state
let gameRunning = false;
let gameLoop;
let animationFrameId;

// Draw functions
function drawRect(x, y, width, height, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, width, height);
}

function drawCircle(x, y, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
}

function drawCenterLine() {
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 15]);
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
}

function clearCanvas() {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Game logic
function update() {
    // Move player paddle
    playerPaddle.y += playerPaddle.dy;
    
    // Keep player paddle in bounds
    if (playerPaddle.y < 0) {
        playerPaddle.y = 0;
    }
    if (playerPaddle.y + playerPaddle.height > canvas.height) {
        playerPaddle.y = canvas.height - playerPaddle.height;
    }
    
    // AI paddle movement
    aiMovement();
    
    // Keep AI paddle in bounds
    if (aiPaddle.y < 0) {
        aiPaddle.y = 0;
    }
    if (aiPaddle.y + aiPaddle.height > canvas.height) {
        aiPaddle.y = canvas.height - aiPaddle.height;
    }
    
    // Move ball
    ball.x += ball.dx;
    ball.y += ball.dy;
    
    // Ball collision with top and bottom
    if (ball.y - ball.size < 0 || ball.y + ball.size > canvas.height) {
        ball.dy = -ball.dy;
    }
    
    // Ball collision with paddles
    if (
        ball.x - ball.size < playerPaddle.x + playerPaddle.width &&
        ball.y > playerPaddle.y &&
        ball.y < playerPaddle.y + playerPaddle.height
    ) {
        ball.dx = -ball.dx;
        // Add some angle based on where ball hits paddle
        const hitPosition = (ball.y - (playerPaddle.y + playerPaddle.height / 2)) / (playerPaddle.height / 2);
        ball.dy = hitPosition * 5;
    }
    
    if (
        ball.x + ball.size > aiPaddle.x &&
        ball.y > aiPaddle.y &&
        ball.y < aiPaddle.y + aiPaddle.height
    ) {
        ball.dx = -ball.dx;
        // Add some angle based on where ball hits paddle
        const hitPosition = (ball.y - (aiPaddle.y + aiPaddle.height / 2)) / (aiPaddle.height / 2);
        ball.dy = hitPosition * 5;
    }
    
    // Scoring
    if (ball.x - ball.size < 0) {
        aiScore++;
        aiScoreElement.textContent = aiScore;
        resetBall();
    }
    
    if (ball.x + ball.size > canvas.width) {
        playerScore++;
        playerScoreElement.textContent = playerScore;
        resetBall();
    }
}

function aiMovement() {
    // Different AI behaviors based on difficulty
    switch(difficulty) {
        case 'easy':
            // Simple follow - moves toward ball but with delay
            aiPaddle.speed = 3;
            if (ball.y < aiPaddle.y + aiPaddle.height / 2) {
                aiPaddle.dy = -aiPaddle.speed;
            } else {
                aiPaddle.dy = aiPaddle.speed;
            }
            // Add some randomness
            if (Math.random() < 0.05) {
                aiPaddle.dy = Math.random() > 0.5 ? aiPaddle.speed : -aiPaddle.speed;
            }
            break;
            
        case 'medium':
            // Predictive movement - tries to position based on ball trajectory
            aiPaddle.speed = 5;
            const targetY = ball.y - aiPaddle.height / 2;
            
            // Predict where ball will be when it reaches AI side
            const timeToReach = (canvas.width - ball.x - ball.size) / Math.abs(ball.dx);
            const predictedY = ball.y + ball.dy * timeToReach;
            
            if (predictedY < aiPaddle.y + aiPaddle.height / 2) {
                aiPaddle.dy = -aiPaddle.speed;
            } else {
                aiPaddle.dy = aiPaddle.speed;
            }
            break;
            
        case 'hard':
            // Advanced predictive with perfect positioning
            aiPaddle.speed = 7;
            const timeToReachHard = (canvas.width - ball.x - ball.size) / Math.abs(ball.dx);
            const predictedYHard = ball.y + ball.dy * timeToReachHard;
            
            // Move directly to predicted position
            if (predictedYHard < aiPaddle.y + aiPaddle.height / 2 - 10) {
                aiPaddle.dy = -aiPaddle.speed;
            } else if (predictedYHard > aiPaddle.y + aiPaddle.height / 2 + 10) {
                aiPaddle.dy = aiPaddle.speed;
            } else {
                aiPaddle.dy = 0;
            }
            break;
    }
    
    aiPaddle.y += aiPaddle.dy;
}

function resetBall() {
    ball.x = canvas.width / 2;
    ball.y = canvas.height / 2;
    ball.dx = Math.random() > 0.5 ? 5 : -5;
    ball.dy = Math.random() * 6 - 3;
}

function draw() {
    clearCanvas();
    drawCenterLine();
    
    // Draw paddles
    drawRect(playerPaddle.x, playerPaddle.y, playerPaddle.width, playerPaddle.height, '#4CAF50');
    drawRect(aiPaddle.x, aiPaddle.y, aiPaddle.width, aiPaddle.height, '#f44336');
    
    // Draw ball
    drawCircle(ball.x, ball.y, ball.size, '#FFF');
}

function gameLoopFunc() {
    if (gameRunning) {
        update();
        draw();
        animationFrameId = requestAnimationFrame(gameLoopFunc);
    }
}

// Event listeners
startBtn.addEventListener('click', () => {
    if (!gameRunning) {
        gameRunning = true;
        gameLoopFunc();
    }
});

resetBtn.addEventListener('click', () => {
    playerScore = 0;
    aiScore = 0;
    playerScoreElement.textContent = playerScore;
    aiScoreElement.textContent = aiScore;
    
    playerPaddle.y = canvas.height / 2 - paddleHeight / 2;
    aiPaddle.y = canvas.height / 2 - paddleHeight / 2;
    
    resetBall();
    
    if (gameRunning) {
        cancelAnimationFrame(animationFrameId);
    }
    gameRunning = false;
    
    clearCanvas();
    drawCenterLine();
    draw();
});

difficultyBtn.addEventListener('click', () => {
    switch(difficulty) {
        case 'easy':
            difficulty = 'medium';
            difficultyBtn.textContent = 'Medium Mode';
            break;
        case 'medium':
            difficulty = 'hard';
            difficultyBtn.textContent = 'Hard Mode';
            break;
        case 'hard':
            difficulty = 'easy';
            difficultyBtn.textContent = 'Easy Mode';
            break;
    }
});

// Keyboard controls
document.addEventListener('keydown', (e) => {
    switch(e.key) {
        case 'w':
        case 'W':
            playerPaddle.dy = -playerPaddle.speed;
            break;
        case 's':
        case 'S':
            playerPaddle.dy = playerPaddle.speed;
            break;
    }
});

document.addEventListener('keyup', (e) => {
    switch(e.key) {
        case 'w':
        case 'W':
        case 's':
        case 'S':
            playerPaddle.dy = 0;
            break;
    }
});

// Initial draw
clearCanvas();
drawCenterLine();
draw();