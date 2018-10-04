using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using TaxiFarePrediction.Models;

namespace TaxiFarePrediction
{
    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static async Task Main(string[] args)
        {
            /* BEGIN This code produces a model and performs a prediction 
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
            Evaluate(model);
            TaxiTripFarePrediction prediction = model.Predict(getTrip());
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);
            */

            /* This code performs a prediction from the model file (Model.zip) */
            TaxiTripFarePrediction prediction = await Predict(getTrip());
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);

        }

        private static TaxiTrip getTrip() {
            return new TaxiTrip() {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripDistance = 10.33f,
                PaymentType = "CSH",
                FareAmount = 0 // predict it. actual = 29.5
            };
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','));
            pipeline.Add(new ColumnCopier(("FareAmount", "Label")));
            pipeline.Add(new CategoricalOneHotVectorizer("VendorId", "RateCode", "PaymentType"));

            pipeline.Add(new ColumnConcatenator("Features",
                                    "VendorId",
                                    "RateCode",
                                    "PassengerCount",
                                    "TripDistance",
                                    "PaymentType"));

            pipeline.Add(new FastTreeRegressor());

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();

            await model.WriteAsync(_modelpath);
            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }

        public static async Task<TaxiTripFarePrediction> Predict(TaxiTrip trip)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> _model = await PredictionModel.ReadAsync<TaxiTrip, TaxiTripFarePrediction>(_modelpath);
            var prediction = _model.Predict(trip);
            return prediction;
        }

    }
}
