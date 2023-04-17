import download_model


class TestHandle:
    def test_g14_download(self, capsys):
        g14_option = "1"
        ret = download_model.handle_key(g14_option)
        stdout, stderr = capsys.readouterr()
        assert "Downloading model..." in stdout
        assert "Download complete!" in stdout
        assert ret == True

    def test_H14_download(self, capsys):
        H14_option = "2"
        ret = download_model.handle_key(H14_option)
        stdout, stderr = capsys.readouterr()
        assert "Downloading model..." in stdout
        assert "Download complete!" in stdout
        assert ret == True

    def test_L14_download(self, capsys):
        L14_option = "3"
        ret = download_model.handle_key(L14_option)
        stdout, stderr = capsys.readouterr()
        assert "Downloading model..." in stdout
        assert "Download complete!" in stdout
        assert ret == True

    def test_B16_download(self, capsys):
        B16_option = "4"
        ret = download_model.handle_key(B16_option)
        stdout, stderr = capsys.readouterr()
        assert "Downloading model..." in stdout
        assert "Download complete!" in stdout
        assert ret == True

    def test_return_to_menu(self, capsys):
        m_option = "m"
        ret = download_model.handle_key(m_option)
        stdout, stderr = capsys.readouterr()
        assert "Returning to menu." in stdout
        assert ret == False
