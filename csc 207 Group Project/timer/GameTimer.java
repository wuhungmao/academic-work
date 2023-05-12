package timer;

import javafx.scene.control.TextField;

import java.util.Timer;
import java.util.TimerTask;

public class GameTimer {
    TextField minutes, seconds;

    int ticks; // Seconds to tick

    private Timer ticker;

    private final TextField toLock;

    public GameTimer(TextField min, TextField sec, TextField input, int time) { // Constructor
        minutes = min;
        seconds = sec;
        toLock = input;
        this.ticks = time;
        input.setDisable(false);
        start(); // Start ticking
    }

    /**
     * Start/Continue ticking
     */
    public void start() {
        if (ticks <= 0) return;
        ticker = new Timer();
        ticker.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (ticks <= 0) { // Time's out
                    ticker.cancel();
                    toLock.setDisable(true); // No input anymore
                    ticker.purge();
                }
                seconds.setText(String.valueOf(ticks % 60)); // Display remaining time
                minutes.setText(String.valueOf(ticks / 60));
                ticks--;
            }
        }, 0, 1000);
    }

    /**
     * Pause the timer
     */
    public void stop() {
        ticker.cancel();
        ticker.purge();
    }
}
