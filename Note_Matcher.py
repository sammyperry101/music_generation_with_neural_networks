import pandas as pd

# Class which matches based on notes in song passed and notes in key of song
class NoteMatcher:

    # Initialise by creating arrays containing each midi value for each potential note
    def __init__(self):
        super().__init__()

        self.c_notes = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
        self.c_sharp_d_flat_notes = [1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121]
        self.d_notes = [2, 14, 26, 38, 50, 62, 74, 86, 98, 110, 122]
        self.d_sharp_e_flat_notes = [3, 15, 27, 39, 51, 63, 75, 87, 99, 111, 123]
        self.e_notes = [4, 16, 28, 40, 52, 64, 76, 88, 100, 112, 124]
        self.f_notes = [5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125]
        self.f_sharp_g_flat_notes = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126]
        self.g_notes = [7, 19, 31, 43, 55, 67, 79, 91, 103, 115, 127]
        self.g_sharp_a_flat_notes = [8, 20, 32, 44, 56, 68, 80, 92, 104, 116]
        self.a_notes = [9, 21, 33, 45, 57, 69, 81, 93, 105, 117]
        self.a_sharp_b_flat_notes = [10, 22, 34, 46, 58, 70, 82, 94, 106, 118]
        self.b_notes = [11, 23, 35, 47, 59, 71, 83, 95, 107, 119]

    # Return all notes in selected keys
    def getCKeyMajor(self):
        return sorted(
            self.c_notes
            + self.d_notes
            + self.e_notes
            + self.f_notes
            + self.g_notes
            + self.a_notes
            + self.b_notes
        )

    def getCKeyMinor(self):
        return sorted(
            self.c_notes
            + self.d_notes
            + self.d_sharp_e_flat_notes
            + self.f_notes
            + self.g_notes
            + self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
        )

    def getCSharpDFlatKeyMajor(self):
        return sorted(
            self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
            + self.f_notes
            + self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
            + self.c_notes
        )

    def getCSharpDFlatKeyMinor(self):
        return sorted(
            self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
            + self.e_notes
            + self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
            + self.a_notes
            + self.b_notes
        )

    def getDKeyMajor(self):
        return sorted(
            self.d_notes
            + self.e_notes
            + self.f_sharp_g_flat_notes
            + self.g_notes
            + self.a_notes
            + self.b_notes
            + self.c_sharp_d_flat_notes
        )

    def getDKeyMinor(self):
        return sorted(
            self.d_notes
            + self.e_notes
            + self.f_notes
            + self.g_notes
            + self.a_notes
            + self.a_sharp_b_flat_notes
            + self.c_notes
        )

    def getDSharpEFlatKeyMajor(self):
        return sorted(
            self.d_sharp_e_flat_notes
            + self.f_notes
            + self.g_notes
            + self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
            + self.b_notes
            + self.c_sharp_d_flat_notes
        )

    def getDSharpEFlatKeyMinor(self):
        return sorted(
            self.d_sharp_e_flat_notes
            + self.f_notes
            + self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
            + self.c_notes
            + self.d_notes
        )

    def getEKeyMajor(self):
        return sorted(
            self.e_notes
            + self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
            + self.a_notes
            + self.b_notes
            + self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
        )

    def getEKeyMinor(self):
        return sorted(
            self.e_notes
            + self.f_sharp_g_flat_notes
            + self.g_notes
            + self.a_notes
            + self.b_notes
            + self.c_notes
            + self.d_notes
        )

    def getFKeyMajor(self):
        return sorted(
            self.f_notes
            + self.g_notes
            + self.a_notes
            + self.a_sharp_b_flat_notes
            + self.c_notes
            + self.d_notes
            + self.e_notes
        )

    def getFKeyMinor(self):
        return sorted(
            self.f_notes
            + self.g_notes
            + self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
            + self.c_notes
            + self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
        )

    def getFSharpGFlatKeyMajor(self):
        return sorted(
            self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
            + self.b_notes
            + self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
            + self.f_notes
        )

    def getFSharpGFlatKeyMinor(self):
        return sorted(
            self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
            + self.a_notes
            + self.b_notes
            + self.c_sharp_d_flat_notes
            + self.d_notes
            + self.e_notes
        )

    def getGKeyMajor(self):
        return sorted(
            self.g_notes
            + self.a_notes
            + self.b_notes
            + self.c_notes
            + self.d_notes
            + self.e_notes
            + self.f_sharp_g_flat_notes
        )

    def getGKeyMinor(self):
        return sorted(
            self.g_notes
            + self.a_notes
            + self.a_sharp_b_flat_notes
            + self.c_notes
            + self.d_notes
            + self.d_sharp_e_flat_notes
            + self.f_notes
        )

    def getGSharpAFlatKeyMajor(self):
        return sorted(
            self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
            + self.c_notes
            + self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
            + self.f_notes
            + self.g_notes
        )

    def getGSharpAFlatKeyMinor(self):
        return sorted(
            self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
            + self.b_notes
            + self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
            + self.e_notes
            + self.f_sharp_g_flat_notes
        )

    def getAKeyMajor(self):
        return sorted(
            self.a_notes
            + self.b_notes
            + self.c_sharp_d_flat_notes
            + self.d_notes
            + self.e_notes
            + self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
        )

    def getAKeyMinor(self):
        return sorted(
            self.a_notes
            + self.b_notes
            + self.c_notes
            + self.d_notes
            + self.e_notes
            + self.f_notes
            + self.g_notes
        )

    def getASharpBFlatKeyMajor(self):
        return sorted(
            self.a_sharp_b_flat_notes
            + self.c_notes
            + self.d_notes
            + self.d_sharp_e_flat_notes
            + self.f_notes
            + self.g_notes
            + self.a_notes
        )

    def getASharpBFlatKeyMinor(self):
        return sorted(
            self.a_sharp_b_flat_notes
            + self.c_notes
            + self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
            + self.f_notes
            + self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
        )

    def getBKeyMajor(self):
        return sorted(
            self.b_notes
            + self.c_sharp_d_flat_notes
            + self.d_sharp_e_flat_notes
            + self.e_notes
            + self.f_sharp_g_flat_notes
            + self.g_sharp_a_flat_notes
            + self.a_sharp_b_flat_notes
        )

    def getBKeyMinor(self):
        return sorted(
            self.b_notes
            + self.c_sharp_d_flat_notes
            + self.d_notes
            + self.e_notes
            + self.f_sharp_g_flat_notes
            + self.g_notes
            + self.a_notes
        )

    # Select which major key to return based on key passed to function
    def getMajorKey(self, key: str):
        if key == "C":
            return self.getCKeyMajor()
        elif key == "C#" or key == "D♭":
            return self.getCSharpDFlatKeyMajor()
        elif key == "D":
            return self.getDKeyMajor()
        elif key == "D#" or key == "E♭":
            return self.getDSharpEFlatKeyMajor()
        elif key == "E":
            return self.getEKeyMajor()
        elif key == "F":
            return self.getFKeyMajor()
        elif key == "F#" or key == "G♭":
            return self.getFSharpGFlatKeyMajor()
        elif key == "G":
            return self.getGKeyMajor()
        elif key == "G#" or key == "A♭":
            return self.getGSharpAFlatKeyMajor()
        elif key == "A":
            return self.getAKeyMajor()
        elif key == "A#" or key == "B♭":
            return self.getASharpBFlatKeyMajor()
        elif key == "B":
            return self.getBKeyMajor()
        else:
            print("ERROR: Key not defined")
            return 0

    # Select which minor key to return based on key passed to function
    def getMinorKey(self, key: str):
        if key == "C":
            return self.getCKeyMinor()
        elif key == "C#" or key == "D♭":
            return self.getCSharpDFlatKeyMinor()
        elif key == "D":
            return self.getDKeyMinor()
        elif key == "D#" or key == "E♭":
            return self.getDSharpEFlatKeyMinor()
        elif key == "E":
            return self.getEKeyMinor()
        elif key == "F":
            return self.getFKeyMinor()
        elif key == "F#" or key == "G♭":
            return self.getFSharpGFlatKeyMinor()
        elif key == "G":
            return self.getGKeyMinor()
        elif key == "G#" or key == "A♭":
            return self.getGSharpAFlatKeyMinor()
        elif key == "A":
            return self.getAKeyMinor()
        elif key == "A#" or key == "B♭":
            return self.getASharpBFlatKeyMinor()
        elif key == "B":
            return self.getBKeyMinor()
        else:
            print("ERROR: Key not defined")
            return 0

    # Select if choosing major or minor key and which key from that, and return all notes in this key
    def getKey(self, key: str):
        keyInfo = key.split(" ")
        key = keyInfo[0]
        majorMinor = keyInfo[1]

        if majorMinor == "Major":
            return self.getMajorKey(key)
        elif majorMinor == "Minor":
            return self.getMinorKey(key)
        else:
            print("ERROR: Key not defined")
            return 0

    # Return the percentage of generated notes that match with the key of the song
    def matchNotes(self, key: str, notes: pd.Series):
        notes_in_key = self.getKey(key)

        num_notes_in_key = 0
        num_notes = 0

        for note in notes:
            if note in notes_in_key:
                print("IN KEY")
                print(note)
                num_notes_in_key += 1
            else:
                print("NOT IN KEY")
                print(note)
            num_notes += 1

        return num_notes_in_key / num_notes
