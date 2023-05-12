package extraEventHandler;

import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Node;
import javafx.scene.control.Label;
import views.BoggleView;

import java.util.ArrayList;

public class BoggleEventHandler implements EventHandler<ActionEvent> {
    private BoggleView view;
    public BoggleEventHandler(BoggleView view) {
        this.view = view;
    }


    @Override
    public void handle(ActionEvent e) {
        ArrayList<String> backupcolorlist = (ArrayList<String>) view.colorList.clone();
        String selectedItem = this.view.colorBlind.getValue();
        if (selectedItem.equals("Red green blindness")) {
            view.colorList.remove("red");
            view.colorList.remove("green");
            view.colorList.remove("yellow");
            int index = 0;
            for (Node letter:this.view.gridpane.getChildren()) {
                if (index == view.colorList.size()) {
                    index = 0;
                }
                Label l = (Label) letter;
                l.setStyle("-fx-text-fill: " + view.colorList.get(index));
                index++;
            }
        } else if (selectedItem.equals("Red blindness")) {
            view.colorList.remove("red");
            int index = 0;
            for (Node letter:this.view.gridpane.getChildren()) {
                if (index == view.colorList.size()) {
                    index = 0;
                }
                Label l = (Label) letter;
                l.setStyle("-fx-text-fill: " + view.colorList.get(index));
                index++;
            }
        } else {
            view.colorList.remove("green");
            int index = 0;
            for (Node letter:this.view.gridpane.getChildren()) {
                if (index == view.colorList.size()) {
                    index = 0;
                }
                Label l = (Label) letter;
                l.setStyle("-fx-text-fill: " + view.colorList.get(index));
                index++;
            }
        }
        view.colorList = backupcolorlist;

    }
}