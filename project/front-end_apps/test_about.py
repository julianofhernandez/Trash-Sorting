import about


class TestHandle:
    def test_key_invalid(self, capsys):
        invalid_option = -1
        ret = about.handle_key(invalid_option)
        stdout, stderr = capsys.readouterr()
        assert "ERROR: Invalid option" in stdout
        assert ret == True

    def test_key_about(self, capsys):
        about_option = "1"
        ret = about.handle_key(about_option)
        stdout, stderr = capsys.readouterr()
        assert about.about_info in stdout
        assert ret == True

    def test_key_menu(self, capsys):
        menu_option = "2"
        ret = about.handle_key(menu_option)
        stdout, stderr = capsys.readouterr()
        assert about.menu_info in stdout
        assert ret == True

    def test_key_credit(self, capsys):
        credit_option = "3"
        ret = about.handle_key(credit_option)
        stdout, stderr = capsys.readouterr()
        assert about.credit_info in stdout
        assert ret == True

    def test_key_exit(self, capsys):
        exit_option = "M"
        ret = about.handle_key(exit_option)
        stdout, stderr = capsys.readouterr()
        assert "Returning to menu" in stdout
        assert ret == False
