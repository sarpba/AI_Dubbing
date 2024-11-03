import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QHBoxLayout, QSplitter, QListWidget, QListWidgetItem, QTableWidget, 
    QTableWidgetItem, QAbstractItemView, QMessageBox, QLineEdit, QSlider,
    QInputDialog
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QBrush, QColor, QFont

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Videólejátszó Felirattal")
        self.setGeometry(100, 100, 1200, 800)

        # Színkódok a beszélőknek
        self.speaker_colors = {
            "SPEAKER_14": "red",
            "SPEAKER_18": "blue",
            "SPEAKER_00": "green",
            "SPEAKER_15": "purple",
            "Unknown": "gray"
            # További beszélők színei
        }

        # Fő Layout
        main_layout = QHBoxLayout()

        # Video és Felirat Layout (Bal oldali panel)
        video_layout = QVBoxLayout()
        self.video_widget = QVideoWidget()
        video_layout.addWidget(self.video_widget)

        # Gombok Layout
        button_layout = QVBoxLayout()
        self.load_button = QPushButton("Videó Betöltése")
        self.load_button.clicked.connect(self.load_video)
        button_layout.addWidget(self.load_button)

        self.play_button = QPushButton("Lejátszás")
        self.play_button.clicked.connect(self.play_video)
        self.play_button.setEnabled(False)
        button_layout.addWidget(self.play_button)

        video_layout.addLayout(button_layout)

        # Felirat Label
        self.subtitle_label = QLabel("")
        self.subtitle_label.setStyleSheet("background-color: rgba(0, 0, 0, 128); color: white; font-size: 16px;")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setFixedHeight(50)
        video_layout.addWidget(self.subtitle_label)

        # Csúszka Létrehozása
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)  # Inicializáláskor a tartomány 0
        self.position_slider.sliderMoved.connect(self.set_position)
        video_layout.addWidget(self.position_slider)

        # Időjelzők Layout
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00:00")
        self.total_time_label = QLabel("00:00:00")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        video_layout.addLayout(time_layout)

        # Média lejátszó
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.positionChanged.connect(self.update_subtitle)

        # Feliratok
        self.subtitles = []  # List of segments
        self.current_subtitle_index = 0  # Kezdetben az első szegmens

        # Jelenlegi kiemelt szegmens indexének nyomon követése
        self.current_highlighted_index = -1  # Kezdetben nincs kiemelt szegmens

        # Jobboldali Szerkesztő Panel
        editor_layout = QVBoxLayout()

        # Lista a szegmensek megjelenítésére
        self.segment_list = QListWidget()
        self.segment_list.itemClicked.connect(self.display_segment_details)
        editor_layout.addWidget(QLabel("Felirat Szegmensek:"))
        editor_layout.addWidget(self.segment_list)

        # **Áthelyezés Az Előző Szegmens Végére Gomb Elhelyezése**
        self.move_to_previous_button = QPushButton("Áthelyezés az előző szegmens végére")
        self.move_to_previous_button.clicked.connect(self.move_words_to_previous_segment_end)
        editor_layout.addWidget(self.move_to_previous_button)

        # Táblázat a szószegmensekhez
        self.word_table = QTableWidget()
        self.word_table.setColumnCount(5)
        self.word_table.setHorizontalHeaderLabels(["Word", "Start (s)", "End (s)", "Score", "Speaker"])
        # Engedélyezzük a Start, End és Word oszlopok szerkesztését, a többit nem
        self.word_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        self.word_table.setSelectionBehavior(QAbstractItemView.SelectRows)  # Teljes sorok kiválasztása
        self.word_table.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Több sor kiválasztása
        editor_layout.addWidget(QLabel("Szószegmensek:"))
        editor_layout.addWidget(self.word_table)

        # Csatlakozzunk a cella módosításához
        self.word_table.cellChanged.connect(self.handle_cell_changed)

        # Beszélő Átnevezése
        speaker_layout = QHBoxLayout()
        self.speaker_input = QLineEdit()
        self.speaker_input.setPlaceholderText("Új Beszélő Név")
        self.rename_button = QPushButton("Beszélő Átnevezése")
        self.rename_button.clicked.connect(self.rename_speaker)
        speaker_layout.addWidget(self.speaker_input)
        speaker_layout.addWidget(self.rename_button)
        editor_layout.addLayout(speaker_layout)

        # **"Áthelyezés a következő szegmens elejére" gomb marad a helyén**
        self.move_to_next_button = QPushButton("Áthelyezés a következő szegmens elejére")
        self.move_to_next_button.clicked.connect(self.move_words_to_next_segment_start)
        editor_layout.addWidget(self.move_to_next_button)

        # Előző és Következő Mondat QLabel-ek
        self.previous_label = QLabel("Előző Mondat: ")
        self.next_label = QLabel("Következő Mondat: ")
        editor_layout.addWidget(self.previous_label)
        editor_layout.addWidget(self.next_label)

        # Mentés Gomb
        self.save_button = QPushButton("Felirat Mentése")
        self.save_button.clicked.connect(self.save_subtitles)
        editor_layout.addWidget(self.save_button)

        # Splitter beállítása
        splitter = QSplitter(Qt.Horizontal)
        video_container = QWidget()
        video_container.setLayout(video_layout)
        editor_container = QWidget()
        editor_container.setLayout(editor_layout)
        splitter.addWidget(video_container)
        splitter.addWidget(editor_container)
        splitter.setSizes([800, 400])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Jelző, hogy a cella frissítése blokkolva van (programozott frissítésekhez)
        self.block_cell_change = False

    def load_video(self):
        # Videó kiválasztása
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Videó Kiválasztása", "", "Videófájlok (*.mp4 *.avi *.mkv)"
        )
        if video_path:
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            self.play_button.setEnabled(True)

            # Felirat fájl keresése (ugyanazzal a névvel, .json kiterjesztéssel)
            json_path = os.path.splitext(video_path)[0] + ".json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.subtitles = data.get("segments", [])
                    print(f"Feliratok betöltve: {json_path}")
                except json.JSONDecodeError as e:
                    QMessageBox.critical(self, "Hiba", f"A felirat fájl nem érvényes JSON: {e}")
                    self.subtitles = []
            else:
                self.subtitles = []
                print("Nincs felirat fájl a kiválasztott videóhoz.")

            self.current_subtitle_index = 0  # Reseteljük az aktuális szegmens indexét
            self.subtitle_label.setText("")
            self.unhighlight_previous_segment()
            self.populate_segment_list()

    def populate_segment_list(self):
        self.segment_list.clear()
        for idx, segment in enumerate(self.subtitles):
            text = segment.get('text', '').strip()
            item = QListWidgetItem(f"{idx + 1}. {text}")
            item.setData(Qt.UserRole, idx)
            # Színkódolás beszélők szerint
            speaker = segment.get('speaker', 'Unknown')
            color = self.speaker_colors.get(speaker, "gray")
            item.setForeground(QBrush(QColor(color)))
            self.segment_list.addItem(item)

    def play_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Lejátszás")
        else:
            self.media_player.play()
            self.play_button.setText("Szünet")

    def update_position(self, position):
        # Frissíti a csúszka értékét a videó aktuális pozíciójának megfelelően
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)
        # Frissíti a jelenlegi időt
        self.current_time_label.setText(self.format_time(position))

    def update_duration(self, duration):
        # Beállítja a csúszka tartományát a videó hosszára
        self.position_slider.setRange(0, duration)
        # Beállítja a teljes időt
        self.total_time_label.setText(self.format_time(duration))

    def set_position(self, position):
        # Beállítja a videó pozícióját a csúszka értékének megfelelően
        self.media_player.setPosition(position)
        # Frissíti a jelenlegi időt
        self.current_time_label.setText(self.format_time(position))

    def format_time(self, ms):
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"

    def update_subtitle(self, position):
        # Position millisec-ben
        position_sec = position / 1000.0

        if not self.subtitles:
            self.subtitle_label.setText("")
            self.unhighlight_previous_segment()
            return

        # Ellenőrizzük, hogy a jelenlegi pozíció beleesik-e egy felirat szegmensbe
        while (self.current_subtitle_index < len(self.subtitles) and
               position_sec > self.subtitles[self.current_subtitle_index].get("end", 0)):
            self.current_subtitle_index += 1

        if self.current_subtitle_index < len(self.subtitles):
            segment = self.subtitles[self.current_subtitle_index]
            if segment.get("start", 0) <= position_sec <= segment.get("end", 0):
                text = segment.get('text', '').strip()
                self.subtitle_label.setText(f"{self.current_subtitle_index + 1}. {text}")
                
                # Kiemeljük a jelenlegi szegmenst a listában
                self.highlight_current_segment()
            else:
                self.subtitle_label.setText("")
                # Ha nincs aktuális szegmens, eltávolítjuk a korábbi kiemelést
                self.unhighlight_previous_segment()
        else:
            self.subtitle_label.setText("")
            self.unhighlight_previous_segment()

    def highlight_current_segment(self):
        # Eltávolítjuk a korábbi kiemelést
        if self.current_highlighted_index != -1:
            previous_item = self.segment_list.item(self.current_highlighted_index)
            if previous_item:
                previous_item.setBackground(QBrush(QColor("white")))  # Alap háttérszín

        # Kiemeljük az új szegmenst
        current_item = self.segment_list.item(self.current_subtitle_index)
        if current_item:
            current_item.setBackground(QBrush(QColor("#ADD8E6")))  # Világoskék háttérszín
            self.current_highlighted_index = self.current_subtitle_index
            # Görgetés a jelenlegi szegmenshez
            self.segment_list.scrollToItem(current_item, QAbstractItemView.PositionAtCenter)

    def unhighlight_previous_segment(self):
        if self.current_highlighted_index != -1:
            previous_item = self.segment_list.item(self.current_highlighted_index)
            if previous_item:
                previous_item.setBackground(QBrush(QColor("white")))  # Alap háttérszín
            self.current_highlighted_index = -1

    def display_segment_details(self, item):
        idx = item.data(Qt.UserRole)
        if idx is not None:
            segment = self.subtitles[idx]
            words = segment.get("words", [])
            self.populate_word_table(idx, words)
            self.display_previous_next_segments(idx)

    def populate_word_table(self, segment_idx, words):
        self.word_table.blockSignals(True)  # Blokkoljuk a jeleket a programozott frissítésekhez
        self.word_table.setRowCount(len(words))
        for row, word in enumerate(words):
            # Word
            word_item = QTableWidgetItem(word.get("word", "").strip())
            # Engedélyezzük a szerkesztést
            word_item.setFlags(word_item.flags() | Qt.ItemIsEditable)
            self.word_table.setItem(row, 0, word_item)
            
            # Start (s)
            start_item = QTableWidgetItem(f"{word.get('start', 0):.3f}")
            start_item.setData(Qt.UserRole, (segment_idx, row))  # Társítsuk a szegmens és szó indexet
            self.word_table.setItem(row, 1, start_item)
            
            # End (s)
            end_item = QTableWidgetItem(f"{word.get('end', 0):.3f}")
            end_item.setData(Qt.UserRole, (segment_idx, row))  # Társítsuk a szegmens és szó indexet
            self.word_table.setItem(row, 2, end_item)
            
            # Score
            score_item = QTableWidgetItem(f"{word.get('score', 0):.3f}")
            score_item.setFlags(score_item.flags() ^ Qt.ItemIsEditable)  # Nem szerkeszthető
            self.word_table.setItem(row, 3, score_item)
            
            # Speaker
            speaker_item = QTableWidgetItem(word.get("speaker", "Unknown"))
            speaker_item.setFlags(speaker_item.flags() ^ Qt.ItemIsEditable)  # Nem szerkeszthető
            self.word_table.setItem(row, 4, speaker_item)

            # Társítsuk a szegmens indexét és a szó indexét a sorhoz
            # Formátum: (segment_index, word_index)
            self.word_table.item(row, 0).setData(Qt.UserRole, (segment_idx, row))

        self.word_table.blockSignals(False)  # Visszakapcsoljuk a jeleket

        # Beállítjuk a sortöltő stílust a jobb olvashatóság érdekében
        self.word_table.setAlternatingRowColors(True)
        self.word_table.setStyleSheet("alternate-background-color: #f0f0f0;")

    def display_previous_next_segments(self, current_idx):
        previous_text = ""
        next_text = ""
        if current_idx > 0:
            previous_text = self.subtitles[current_idx - 1].get("text", "").strip()
        if current_idx < len(self.subtitles) - 1:
            next_text = self.subtitles[current_idx + 1].get("text", "").strip()
        self.previous_label.setText(f"Előző Mondat: {previous_text}")
        self.next_label.setText(f"Következő Mondat: {next_text}")

    def rename_speaker(self):
        new_speaker = self.speaker_input.text().strip()
        if not new_speaker:
            QMessageBox.warning(self, "Figyelmeztetés", "Add meg az új beszélő nevét.")
            return

        selected_items = self.segment_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Figyelmeztetés", "Válassz ki egy felirat szegmenst a beszélő átnevezéséhez.")
            return

        item = selected_items[0]
        idx = item.data(Qt.UserRole)
        if idx is not None:
            old_speaker = self.subtitles[idx].get("speaker", "Unknown")
            # Átnevezés a segments részben
            self.subtitles[idx]["speaker"] = new_speaker
            # Átnevezés a szegmens belső szavainak részben
            for word in self.subtitles[idx].get("words", []):
                if word.get("speaker", "Unknown") == old_speaker:
                    word["speaker"] = new_speaker
            # Frissítjük a lista megjelenítését
            text = self.subtitles[idx].get('text', '').strip()
            item.setText(f"{idx + 1}. {text}")
            # Frissítjük a színkódolást
            color = self.speaker_colors.get(new_speaker, "gray")
            item.setForeground(QBrush(QColor(color)))
            QMessageBox.information(self, "Siker", f"A beszélő átnevezve {new_speaker}-re.")
            self.speaker_input.clear()

    def move_words_to_previous_segment_end(self):
        selected_items = self.word_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Figyelmeztetés", "Válassz ki egy vagy több szót a táblázatból.")
            return

        # Azonosítsd a kiválasztott sorokat és gyűjtsd össze a szegmens és szó indexeket
        selected_rows = set()
        for item in selected_items:
            selected_rows.add(item.row())

        # Fontos: Iteráljunk csökkenő sorrendben, hogy a törlés ne zavarja az indexeket
        selected_word_indices = sorted([self.word_table.item(row, 0).data(Qt.UserRole) for row in selected_rows], reverse=True)

        # Kiválasztott szavak áthelyezése
        moved_words = []
        affected_segments = set()
        for (segment_idx, word_idx) in selected_word_indices:
            if segment_idx == 0:
                QMessageBox.warning(self, "Hiba", f"A szó '{self.subtitles[segment_idx]['words'][word_idx]['word']}' nem mozdítható az előző szegmens végére, mivel nincs előző szegmens.")
                continue
            previous_segment = self.subtitles[segment_idx - 1]
            current_segment = self.subtitles[segment_idx]
            word = current_segment['words'][word_idx]

            # Áthelyezés előző szegmens végére
            previous_segment['words'].append(word)
            # Rendezés az előző szegmensben az start időpont alapján
            previous_segment['words'].sort(key=lambda w: w['start'])

            # Frissítsük az előző szegmens end időpontját a szó end időpontjára
            previous_segment['end'] = word['end']

            # Szó törlése a jelenlegi szegmensből
            del current_segment['words'][word_idx]

            moved_words.append(word['word'])
            affected_segments.add(segment_idx - 1)
            affected_segments.add(segment_idx)

        if moved_words:
            QMessageBox.information(self, "Siker", f"A szavak sikeresen áthelyezve az előző szegmens végére:\n{', '.join(moved_words)}")

        # Frissítjük a szegmensek 'text' mezőit és időintervallumait
        for seg_idx in affected_segments:
            self.update_segment_text(seg_idx)

        # Frissítjük a szegmensek listáját és a jelenlegi szegmens részleteit
        self.populate_segment_list()
        self.display_segment_details(self.segment_list.item(self.current_subtitle_index))

    def move_words_to_next_segment_start(self):
        selected_items = self.word_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Figyelmeztetés", "Válassz ki egy vagy több szót a táblázatból.")
            return

        # Azonosítsd a kiválasztott sorokat és gyűjtsd össze a szegmens és szó indexeket
        selected_rows = set()
        for item in selected_items:
            selected_rows.add(item.row())

        # Az 'Word' oszlopban van a UserRole, tehát onnan kell lekérdezni
        selected_word_indices = sorted([self.word_table.item(row, 0).data(Qt.UserRole) for row in selected_rows])

        # Kiválasztott szavak áthelyezése
        moved_words = []
        affected_segments = set()
        for (segment_idx, word_idx) in selected_word_indices:
            if segment_idx >= len(self.subtitles) - 1:
                QMessageBox.warning(self, "Hiba", f"A szó '{self.subtitles[segment_idx]['words'][word_idx]['word']}' nem mozdítható a következő szegmens elejére, mivel nincs következő szegmens.")
                continue
            next_segment = self.subtitles[segment_idx + 1]
            current_segment = self.subtitles[segment_idx]
            word = current_segment['words'][word_idx]

            # Áthelyezés következő szegmens elejére
            next_segment['words'].insert(0, word)
            # Rendezés a következő szegmensben az start időpont alapján
            next_segment['words'].sort(key=lambda w: w['start'])

            # Frissítsük a következő szegmens start időpontját a szó start időpontjára
            next_segment['start'] = word['start']

            # Szó törlése a jelenlegi szegmensből
            del current_segment['words'][word_idx]

            moved_words.append(word['word'])
            affected_segments.add(segment_idx + 1)
            affected_segments.add(segment_idx)

        if moved_words:
            QMessageBox.information(self, "Siker", f"A szavak sikeresen áthelyezve a következő szegmens elejére:\n{', '.join(moved_words)}")

        # Frissítjük a szegmensek 'text' mezőit és időintervallumait
        for seg_idx in affected_segments:
            self.update_segment_text(seg_idx)

        # Frissítjük a szegmensek listáját és a jelenlegi szegmens részleteit
        self.populate_segment_list()
        self.display_segment_details(self.segment_list.item(self.current_subtitle_index))

    def update_segment_text(self, segment_idx):
        """
        Frissíti a megadott szegmens 'text' mezőjét a belső 'words' listából.
        """
        if 0 <= segment_idx < len(self.subtitles):
            words = self.subtitles[segment_idx].get('words', [])
            # Összeállítjuk a 'text' mezőt a szavakból
            text = ' '.join([word.get('word', '').strip() for word in words])
            self.subtitles[segment_idx]['text'] = text
            print(f"Szegmens {segment_idx + 1} text frissítve: {text}")
        else:
            print(f"Hibás szegmens index: {segment_idx}")

    def handle_cell_changed(self, row, column):
        if self.block_cell_change:
            return

        # Csak a Word, Start (s) és End (s) oszlopokat kezeljük
        if column not in [0, 1, 2]:
            return

        item = self.word_table.item(row, column)
        if not item:
            return

        if column == 0:
            # Word oszlop
            new_word = item.text().strip()
            if not new_word:
                QMessageBox.warning(self, "Hiba", "A szó nem lehet üres.")
                # Visszaállítjuk az eredeti értéket
                self.block_cell_change = True
                segment_idx, word_idx = item.data(Qt.UserRole)
                original_word = self.subtitles[segment_idx]['words'][word_idx]['word']
                item.setText(original_word)
                self.block_cell_change = False
                return

            # Frissítjük a szegmens 'text' mezőjét
            segment_idx, word_idx = item.data(Qt.UserRole)
            self.subtitles[segment_idx]['words'][word_idx]['word'] = new_word
            self.update_segment_text(segment_idx)
            self.populate_segment_list()
            self.display_segment_details(self.segment_list.item(segment_idx))
            return

        new_value = item.text().strip()
        try:
            new_time = float(new_value)
        except ValueError:
            QMessageBox.warning(self, "Hiba", f"A(z) '{item.text()}' nem érvényes szám az oszlopban.")
            # Visszaállítjuk az eredeti értéket
            self.block_cell_change = True
            segment_idx, word_idx = item.data(Qt.UserRole)
            original_time = self.subtitles[segment_idx]['words'][word_idx]['start'] if column == 1 else self.subtitles[segment_idx]['words'][word_idx]['end']
            item.setText(f"{original_time:.3f}")
            self.block_cell_change = False
            return

        segment_idx, word_idx = item.data(Qt.UserRole)
        if column == 1:
            # Start (s) oszlop
            self.subtitles[segment_idx]['words'][word_idx]['start'] = new_time
        elif column == 2:
            # End (s) oszlop
            self.subtitles[segment_idx]['words'][word_idx]['end'] = new_time

        # Ellenőrizzük, hogy a start kisebb-e az endnél
        word = self.subtitles[segment_idx]['words'][word_idx]
        if word['start'] >= word['end']:
            QMessageBox.warning(self, "Hiba", f"A(z) '{word['word']}' szó start időpontja nem lehet nagyobb vagy egyenlő az end időpontjával.")
            # Visszaállítjuk az eredeti értéket
            self.block_cell_change = True
            original_time = word['start'] if column == 1 else word['end']
            item.setText(f"{original_time:.3f}")
            self.block_cell_change = False
            return

        # Frissítjük a szegmens 'text' mezőjét
        self.update_segment_text(segment_idx)

    def save_subtitles(self):
        # Mentés helyének kiválasztása
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Felirat Mentése", "", "JSON Fájlok (*.json)"
        )
        if save_path:
            data = {
                "segments": self.subtitles
            }
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                QMessageBox.information(self, "Siker", "Feliratok sikeresen mentve.")
            except Exception as e:
                QMessageBox.critical(self, "Hiba", f"Hiba történt a mentés során: {e}")

    def closeEvent(self, event):
        # Mentés előtti megerősítés
        reply = QMessageBox.question(
            self, 'Kilépés', 'Biztosan ki akarsz lépni?', 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
