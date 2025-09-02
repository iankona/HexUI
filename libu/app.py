try:
    from . import tkfunction as UI
    from . import tkfunctionexpand as EX
except:
    import tkfunction as UI
    import tkfunctionexpand as EX




if __name__ == '__main__':
    with UI.Tk():
        with UI.NoteBookPack():
            with EX.CanvasCADPack(), UI.Title("画布1"):
                pass
                # draw()
            EX.TreeProcessPack()
            with UI.CanvasPack(), UI.Title("画布2"):
                pass
                # draw()
            with UI.PanedWindowVPack(), UI.Title("标题1"):
                with UI.FramePack(width=45, height=60, side="top", expand=1):
                    UI.LabelPack(text="鹅鹅鹅1")
                    UI.LabelPack(text="鹅鹅鹅")
                    UI.LabelPack(text="鹅鹅鹅")
                with UI.PanedWindowHPack():
                    with UI.FramePack(width=55, height=70, side="top", expand=1):
                        UI.LabelPack(text="鹅鹅鹅21")
                        UI.LabelPack(text="鹅鹅鹅")
                        UI.LabelPack(text="鹅鹅鹅")
                    with UI.FramePack(width=65, height=80, side="top", expand=1):
                        UI.LabelPack(text="鹅鹅鹅22")
                        UI.LabelPack(text="鹅鹅鹅")
                        UI.LabelPack(text="鹅鹅鹅")
                with UI.FramePack(width=75, height=90, side="top", expand=1):
                    UI.LabelPack(text="鹅鹅鹅3")
                    UI.LabelPack(text="鹅鹅鹅")
                    UI.LabelPack(text="鹅鹅鹅")
            with UI.FramePack(width=450, height=600, side="top", expand=1), UI.Title("标题2"):
                    UI.LabelPack(text="鹅鹅鹅1")
                    UI.LabelPack(text="鹅鹅鹅")
                    UI.LabelPack(text="鹅鹅鹅")