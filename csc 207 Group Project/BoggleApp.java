import views.BoggleView;

import javafx.application.Application;
import javafx.stage.Stage;

public class BoggleApp extends Application {
    BoggleView view;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {
        view = new BoggleView(stage);
    }
}
