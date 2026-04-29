class BetSizingAbstraction:
    """
    Abstracción tipo solver (PioSOLVER / DeepStack idea)

    Convierte apuestas continuas → acciones estratégicas discretas.
    """

    def __init__(self):
        # fracciones típicas de poker solver
        self.sizes = [
            0.5,   # small bet
            1.0,   # pot bet
            2.0    # overbet / pressure
        ]

    # -----------------------------
    # MAIN ACTION GENERATION
    # -----------------------------
    def get_actions(self, pot, stack, min_raise):

        actions = ["fold", "call"]

        for size in self.sizes:

            bet_amount = pot * size

            # clamp por stack
            bet_amount = min(bet_amount, stack)

            # si no es válido, ignorar
            if bet_amount >= min_raise:
                actions.append(f"raise_{size}")

        # siempre permitir all-in estratégico
        actions.append("all_in")

        return actions

    # -----------------------------
    # NORMALIZATION (OPTIONAL)
    # -----------------------------
    def normalize(self, action, pot, stack):

        if action == "all_in":
            return stack

        if action == "call":
            return 1

        if action == "fold":
            return 0

        if "raise_" in action:
            size = float(action.split("_")[1])
            return min(pot * size, stack)

        return 1