#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nshogi/core/initializer.h>
#include <nshogi/io/sfen.h>

#include "types.h"
#include "mcts.h"
#include "nn.h"
#include "encoder.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m) {
        nshogi::core::initializer::initializeAll();
        
        py::class_<MCTS::SearchResult>(m, "SearchResult")
                .def_readonly("best_move", &MCTS::SearchResult::best_move)
                .def_readonly("visit_counts", &MCTS::SearchResult::visit_counts)
                .def_readonly("value", &MCTS::SearchResult::root_value);
        
        py::class_<nshogi::core::Move32>(m, "Move32")
                .def(py::init<>())
                .def("__str__", [](const nshogi::core::Move32& m) {
                        std::stringstream ss;
                        ss << nshogi::io::sfen::move32ToSfen(m); 
                        return ss.str();
                });
        
        py::class_<GameState>(m, "GameState")
        .def("__str__", [](const GameState& state) {
                std::stringstream ss;
                ss << nshogi::io::sfen::stateToSfen(*(state.state)); 
                return ss.str();
        })
        .def_static("initial", []() {
                return GameState(nshogi::io::sfen::StateBuilder::getInitialState());
        })
        .def("is_terminal", &GameState::is_terminal)
        .def("result", &GameState::result)
        .def("legal_moves", &GameState::legal_moves)
        .def("do_move", &GameState::do_move);
        
        py::class_<NeuralNetwork, std::shared_ptr<NeuralNetwork>>(m, "NeuralNetwork")
        .def(py::init<const std::string&, int>(), py::arg("path"), py::arg("queue_size") = 16)
        .def("evaluate", &NeuralNetwork::evaluate_async)
        .def("get_encoded_state", [](NeuralNetwork &self, const GameState &state) {
                auto encoded = encode_state(state); 
                auto arr = py::array_t<float>({48, 9, 9}); // Use 48 from types.h
                std::memcpy(arr.mutable_data(), encoded.data(), encoded.size() * sizeof(float));
                return arr;
        }, py::arg("state"));
        py::class_<NodePool>(m, "NodePool")
                .def(py::init<>())
                .def("reset", &NodePool::reset);
                
        py::class_<MCTS>(m, "MCTS")
                .def(py::init<NodePool&>(), py::arg("pool"))
                .def("start_new_game", &MCTS::start_new_game)
                .def("update_root", &MCTS::update_root)
                .def("search", &MCTS::search, py::arg("state"), py::arg("nn"), py::arg("iterations"),
                py::return_value_policy::reference, // Don't copy the return value
                py::call_guard<py::gil_scoped_release>(),
                "Search the game tree");
}
